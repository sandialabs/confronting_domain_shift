#Based on open source repo:
#    https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py
import torch
import torch.nn as nn
import numpy as np
import time
import math
import os
from matplotlib import pyplot as plt
import pickle
import json
from data_prep import *
from models import WaveNetModel, TransAm
import argparse

def train(train_data, train_lbls, model, batch_size, optimizer, criterion, input_window=128, output_window=1):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, train_lbls, i,batch_size, input_window, output_window)
        if model.name == "wavenet":
            data = torch.transpose(data, 1,0)
            data = torch.transpose(data, 1,2)
            targets = torch.transpose(targets, 1,0)
            targets = torch.transpose(targets, 1,2)
        output = model(data)
        #print("TRAIN OUT", output.size())
        #if model.name == "wavenet":
            #print("OUT", output.size())
            #output = output.unsqueeze(-1)
        #    output = torch.cat((data[:,:,1:], output[:,:,-1:]), dim = -1)
        #predict several steps
        for j in range(output_window-1):
            if model.name == "wavenet":
                data = torch.cat((data, output[:,:,-1:]), dim =-1)
                output = torch.cat((output, model(data[:,:,-input_window:])[:,:,-1:]), dim =-1)
                #output = torch.cat((output, model(data[:,:,-input_window:]).unsqueeze(-1)[:,:,-1:]), dim=-1)
            else:
                data = torch.cat((data, output[-1:]))
                output = torch.cat((output, model(data[-input_window:])[-1:]))
            #print("DATA", data.size())
            #print("OUT", output.size())
            #output = torch.cat((model(data[-input_window:]), output[-1:]))
        optimizer.zero_grad()
        #loss = criterion(output_seg, target_seg)
        #print("TRAIN OUT", output.size(), "TGT", targets[:,:,-output_window:].size())
        if model.name == "wavenet":
            loss = criterion(output, targets[:,:,-output_window:])
        else:
            loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1000#int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, train_data, train_lbls, input_window, output_window, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(train_data) - 1, eval_batch_size):
            data, targets = get_batch(train_data, train_lbls, i,eval_batch_size, input_window, output_window)
            if eval_model.name == "wavenet":
                data = torch.transpose(data, 1,0)
                data = torch.transpose(data, 1,2)
                targets = torch.transpose(targets, 1,0)
                targets = torch.transpose(targets, 1,2)
            output = eval_model(data)[:,:,-1]
                
            if eval_model.name == "wavenet":
                tgt = targets[:,:,input_window-1]
            else:
                tgt = targets[:input_window,:].squeeze()
            #for res in range(len(output)):
            #    print("OUT", output[res], "TGT", tgt[res])
            
            #total_loss += len(data[0])* criterion(output, targets[:len(targets)-(output_window-1)]).cpu().item()
            #print("OUT", output.size(), "TGT", tgt.size())
            total_loss += len(data)* criterion(output, tgt).cpu().item()

    return total_loss# / len(train_data)

def main(cfg_dict):
    torch.manual_seed(cfg_dict["torch_rng_seed"])
    device = torch.device("cuda:{}".format(cfg_dict["gpu_num"]) if torch.cuda.is_available() else "cpu")
    fs = sorted([f for f in os.listdir(cfg_dict["datadir"]) if f.endswith(".csv")])

    model_dir = "./{}_models_{}/".format(cfg_dict["model_type"], cfg_dict["data_name"])
    log_dir = "./{}_logs_{}/".format(cfg_dict["model_type"], cfg_dict["data_name"])
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    for f in fs:
        for output_window in cfg_dict["output_windows"]:
            for input_window in cfg_dict["input_windows"]:
                
                val_losses = []
                print("Initializing {} model".format(cfg_dict["model_type"]))
                if cfg_dict["model_type"] == "wavenet":
                    if input_window == 32:
                        num_layers = 3
                    else:
                        num_layers = 4
                    model = WaveNetModel(classes=1, output_length=1, layers=num_layers).to(device)
                else: #Transformer
                    model = TransAm().to(device)
                datapath = os.path.join(cfg_dict["datadir"],f)
                print("******************Training with {} input {} output {}*********************".format(f, input_window, output_window))
                train_data, train_lbls, val_data, val_lbls, test_data, test_lbls = get_data(datapath, input_window, output_window,downsample_factor=cfg_dict["downsample_factor"], n_train=cfg_dict["n_train"], n_test=cfg_dict["n_test"])
                #downsample_factor=100, n_train=2000, n_test=300)
                train_data = train_data.to(device)
                train_lbls = train_lbls.to(device)
                val_data = val_data.to(device)
                val_lbls = val_lbls.to(device)
                test_data = test_data.to(device)
                test_lbls = test_lbls.to(device)
                
                criterion = nn.MSELoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_dict['lr'])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)

                best_val_loss = float("inf")
                epochs = cfg_dict["num_epochs"] # The number of epochs
                best_model = None

                for epoch in range(1, epochs + 1):
                    epoch_start_time = time.time()
                    train(train_data, train_lbls, model, cfg_dict["batch_size"], optimizer, criterion, input_window=input_window, output_window=output_window)
                    val_loss = evaluate(model, val_data, val_lbls, input_window, output_window, criterion)
                    print("VAL LOSS {}".format(val_loss))
                    val_losses.append(val_loss)
                    #if(epoch % 10 is 0):
                    #    predict_future(model, test_data, test_lbls, input_window,input_window, plot=True)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        print("NEW BEST!")
                        torch.save(best_model.state_dict(), os.path.join(model_dir,"{}_outputwindow_{}_inputwindow_{}.pth".format(f[:-4], output_window, input_window)))
                    scheduler.step() 
                with open(os.path.join(log_dir, "{}_outputwindow_{}_inputwindow_{}.pkl".format(f[:-4], output_window, input_window)), 'wb') as log_f:
                    pickle.dump(val_losses,log_f)

                #predict_future(best_model, test_data, test_lbls, input_window, input_window, plot=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fname', help='path to config file')
    args = parser.parse_args()
    with open(args.config_fname, 'r') as f:
        s = f.read()
    cfg_dict = json.loads(s)
    main(cfg_dict)


