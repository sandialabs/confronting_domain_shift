#Based on open source repo:
#    https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py
import torch
import torch.nn as nn
import numpy as np
import time
import math
import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
import json
from data_prep import *
import argparse
from models import WaveNetModel, TransAm
from torch.nn.functional import mse_loss
import random
random.seed(777)

#Deal with different input/output format for each model
def make_pred(model, data, input_window):
    if model.name == "wavenet":
        data = torch.transpose(data, 1,0)
        data = torch.transpose(data, 1,2)
        output = model(data[:,:,-input_window:])
        data = torch.transpose(data, 1,2)
        data = torch.transpose(data, 1,0)
        output = torch.transpose(output, 1,2)
        output = torch.transpose(output, 1,0)[:,:,-1:]
    else:
        output = model(data[-input_window:])[-1:]
    return output

def prep_combo_data(cfg, model_path):
    input_window = cfg["input_window"]
    train_data, train_lbls = prep_data(cfg, model_path, test_set=True, k=cfg["trainK"], input_window=input_window+cfg["infer_steps"])
    test_data, test_lbls = prep_data(cfg, model_path, test_set=False, k=cfg["testK"])
    closest_pt, test_idx = find_intersection(train_lbls[:,0,0], test_data[:,0,0])
    combo = torch.cat((train_data, train_lbls[:closest_pt], test_data[test_idx:]))
    input_pad = 32 #ensure there is some original domain in prediction seq
    combo = combo[-(input_pad + input_window + cfg["infer_steps"]):-input_pad]

    domain_shift_strt = len(combo) - len(test_data[test_idx:]) + input_pad
    print("COMBO", len(combo))
    return combo[:input_window], combo[input_window: input_window+cfg["infer_steps"]], domain_shift_strt

def prep_data(cfg, model_path, test_set=False, k=None, infer_steps=None, input_window=None, datadir=None):
    if not datadir:
        datadir = cfg["datadir"]
    if not input_window:
        input_window = cfg["input_window"]
    if cfg["data_name"] == "com":
        data_name = model_path[:model_path.find("output")-1]
        datapath = datadir +"chain_of_masses_20210414_k{}_0damp/01dof/{}.csv".format(k,data_name)
    else:#beam
        datapath = datadir +"w.csv"
        
    data = gen_dataset(datapath, downsample_factor=cfg["downsample_factor"])
    if test_set:
        data = torch.FloatTensor(data[-cfg["n_test"]:])
    else:
        data = torch.FloatTensor(data[:cfg["n_train"]])
    
    infer_data = data[:input_window]
    if infer_steps:
        lbls = data[input_window:input_window+infer_steps]
    else:
        lbls = data[input_window:]

    return infer_data.unsqueeze(-1).unsqueeze(-1), lbls.unsqueeze(-1).unsqueeze(-1)

def write_fs(outdir, model_path, pred, mc_preds=None, dsi=None, diffs=None):
    os.makedirs(outdir, exist_ok=True)
    d = {"pred":pred.cpu(), "mc_preds":mc_preds,"dsi":dsi, "diffs":diffs}
    with open(os.path.join(outdir, model_path[:-4]+".pkl"), 'wb') as f:
        pickle.dump(d, f)

def build_model(cfg):
    if cfg["model_type"] == "wavenet":
         if cfg["input_window"] == 32:
             num_layers = 3
         else:
             num_layers = 4
         model = WaveNetModel(classes=1, output_length=1, layers=num_layers, dropout_rate=cfg["infer_dropout_rate"])
    else: #Transformer
         model = TransAm(dropout=cfg["infer_dropout_rate"])
    return model

def correct_pred(cfg, pred, mc_preds, correction_mode, train_unc_max=None, correct_all_preds=False):
    unc = mc_preds.std()

    if correct_all_preds:
        thresh_cond = True
    elif cfg["unc_thresh_mode"] == "train_max":
        thresh_cond = unc > train_unc_max*cfg["unc_thresh"]
    else: #thresh_mode == "ten_pct"
        thresh_cond = unc >= 0.1*pred.item()
        
    if thresh_cond:
        if correction_mode == "mean":
            pred[0,:,:] = mc_preds.mean()
        else: #skew
            preds_gt_output = (1*(mc_preds > pred.item())).sum()
            infer_iters = cfg["infer_iters"]
            if preds_gt_output > infer_iters//2:
                pred = pred + cfg["corr_factor"]*unc
            elif preds_gt_output < infer_iters//2:
                pred = pred - cfg["corr_factor"]*unc
            else: #50/50 split
                pass

    return thresh_cond, pred

def infer_uq(cfg, model, infer_data, infer_lbl, correction_mode=None, train_unc_max=None, correct_all=False):
    torch.manual_seed(cfg["torch_rng_seed"])
    steps = cfg["infer_steps"]
    infer_iters = cfg["infer_iters"]
    mc_preds = np.zeros((steps,infer_iters))
    domain_shift_indicators = []
    pred_diffs = []
    
    with torch.no_grad():
        for i in range(steps):
            if i == 0:
                output = infer_lbl[i].unsqueeze(0)
                mc_preds[i,:] = output.item()
            elif i % cfg["sensor_reading_freq"] == 0:
                output = torch.cat((output, infer_lbl[i].unsqueeze(0)))
                mc_preds[i,:] = output[-1:].item()
            else:
                model.train()
                for j in range(infer_iters):
                    mc_preds[i,j] = make_pred(model, infer_data, cfg["input_window"])
                model.eval()
                pred = make_pred(model, infer_data, cfg["input_window"])
                if correction_mode:
                    correct_all_preds = any(domain_shift_indicators) and correct_all
                    thresh_cond, corrected_pred = correct_pred(cfg, pred, mc_preds[i,:], correction_mode, train_unc_max=train_unc_max, correct_all_preds=correct_all_preds)
                    domain_shift_indicators.append(thresh_cond)
                    pred_diffs.append((pred-corrected_pred).item())
                    pred = corrected_pred
                output = torch.cat((output,pred))
            infer_data = torch.cat((infer_data, output[-1:]))
    return output, mc_preds, domain_shift_indicators, pred_diffs

def infer(cfg, model, infer_data, infer_lbl):
    model.eval()
    with torch.no_grad():
        for i in range(cfg["infer_steps"]):
            if i == 0:
                output = infer_lbl[i].unsqueeze(0)
            elif i % cfg["sensor_reading_freq"] == 0:
                output = torch.cat((output, infer_lbl[i].unsqueeze(0)))
            else:
                output = torch.cat((output,make_pred(model, infer_data, cfg["input_window"])))
            infer_data = torch.cat((infer_data, output[-1:]))
    return output

def main(cfg):
    device = torch.device("cuda:{}".format(cfg["gpu_num"]) if torch.cuda.is_available() else "cpu")
    model_paths = sorted([f for f in os.listdir(cfg["model_dir"]) if f.endswith(".pth")])
    outdir = "./{}_output_{}_mode_{}_testK_{}".format(cfg["model_type"], cfg["data_name"], cfg["unc_thresh_mode"], cfg["testK"])
    correct_all = cfg["correct_all"] if "correct_all" in cfg else False
    if correct_all:
        outdir = outdir + "_correct_all"
    for i in range(len(model_paths)):
        os.makedirs(outdir, exist_ok=True)
        model_path = model_paths[i]
        print("Loading model {}".format(model_path))
        model = build_model(cfg)
        model.load_state_dict(torch.load(os.path.join(cfg["model_dir"], model_path), map_location=device))
        model = model.to(device)
        infer_data, infer_lbl = prep_data(cfg, model_path, test_set=True, k=cfg["trainK"], infer_steps=cfg["infer_steps"])
        infer_data = infer_data.to(device)
        infer_lbl = infer_lbl.to(device)

        #Pass label in to simulate sensor readings
        if cfg["calc_uq"]:
            if cfg["unc_thresh_mode"]=="train_max":
                pred, mc_preds, _, _ = infer_uq(cfg, model, infer_data, infer_lbl)
                train_unc_max = np.max(mc_preds.std(axis=-1))
            else:
                train_unc_max = None
            if cfg["data_name"]=="com":
                infer_data, infer_lbl, domain_shift_idx = prep_combo_data(cfg, model_path)
            else:
                infer_data, infer_lbl  = prep_data(cfg, model_path, test_set=True, k=cfg["trainK"], infer_steps=cfg["infer_steps"], datadir=cfg["infer_path"])
                domain_shift_idx = 0
            out_data_dir = os.path.join(outdir,"data") 
            os.makedirs(out_data_dir, exist_ok=True)
            with open(os.path.join(out_data_dir, "infer_data_{}.pkl".format(model_path[:-4])), 'wb') as f:
                pickle.dump({'data':infer_data,'lbl':infer_lbl, 'dsi':domain_shift_idx}, f)

            infer_data = infer_data.to(device)
            infer_lbl = infer_lbl.to(device)

            orig_pred = infer(cfg, model, infer_data, infer_lbl)
            write_fs(os.path.join(outdir,"orig_preds"), model_path, orig_pred)
            mean_pred, mean_mc_preds, dsi, mean_diffs = infer_uq(cfg, model, infer_data, infer_lbl, correction_mode="mean", train_unc_max=train_unc_max, correct_all=correct_all)
            write_fs(os.path.join(outdir,"mean_preds"), model_path, mean_pred, mc_preds=mean_mc_preds, dsi=dsi, diffs=mean_diffs)
            skew_pred, skew_mc_preds, dsi, skew_diffs = infer_uq(cfg, model, infer_data, infer_lbl, correction_mode="skew", train_unc_max=train_unc_max, correct_all=correct_all)
            write_fs(os.path.join(outdir,"skew_preds"), model_path, skew_pred, mc_preds=skew_mc_preds, dsi=dsi, diffs=skew_diffs)
            print("ORIG MSE:{}".format(mse_loss(orig_pred, infer_lbl)))
            print("MEAN MSE:{}".format(mse_loss(mean_pred, infer_lbl)))
            print("SKEW MSE:{}".format(mse_loss(skew_pred, infer_lbl)))
            
        else:
            pred = infer(cfg, model, infer_data, infer_lbl)
            print("MSE:{}".format(mse_loss(pred, infer_lbl)))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fname', help='path to config file')
    args = parser.parse_args()
    with open(args.config_fname, 'r') as f:
        s = f.read()
    cfg = json.loads(s)
    main(cfg)    
