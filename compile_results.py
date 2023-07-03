import argparse
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def print_stats(method, mse, ds_mse, mape):
    print("{}: MSE {} +/- {}; domain shift MSE {} +/- {}; MAPE {} +/- {}".format(method, np.mean(mse), np.std(mse), np.mean(ds_mse), np.std(ds_mse), np.mean(mape), np.std(mape)))
          

def plot_mses(train_mse, corr_mse, baseline, corr_mode, outdir):
    sorted_idxs = np.argsort(train_mse)
    plt.figure()
    plt.scatter(train_mse[sorted_idxs],corr_mse[sorted_idxs], s=2)
    xs = np.linspace(0, train_mse.max())
    plt.plot(xs, xs, c='black')
    plt.show()
    plt.savefig(os.path.join(outdir, "{}_vs_{}.png".format(corr_mode, baseline)))
    plt.close()
    
def print_results(outdir):
    newdir = os.path.join(outdir, "analysis")
    os.makedirs(newdir, exist_ok=True)
    
    mse_tests = []
    mse_means = []
    mse_skews = []
    mape_tests = []
    mape_means = []
    mape_skews = []
    
    ds_mse_tests = []
    ds_mse_means = []
    ds_mse_skews = []
    ds_results = []
    mape_results = []
    
    ds_detected = 0
    false_pos = 0
    fs = sorted([f for f in os.listdir(outdir) if "mode" in f and "uq" in f and f.endswith(".pkl")])
    mse_f = [f for f in os.listdir(outdir) if "mode" in f and "mse" in f and f.endswith(".pkl")][0]
    with open(os.path.join(outdir, mse_f), 'rb') as f:
        train_d = pickle.load(f)
    train_mses = [e.item() for e in train_d['mses']]

     
    for i in range(len(fs)):
        with open(os.path.join(outdir, fs[i]), 'rb') as f:
            d = pickle.load(f)

        if d["dsi_mean"][-1] !=-1:# >= d["dss"]:
            ds_detected +=1
        if d["dsi_mean"][-1] != -1 and d["dsi_mean"][-1] < d["dss"]:
            false_pos +=1

        mse_tests.append(d["mse_test"].item())
        mse_means.append(d["mse_mean"].item())
        mse_skews.append(d["mse_skew"].item())

        if "mape_test" in d:
            mape_tests.append(d["mape_test"].item())
            mape_means.append(d["mape_mean"].item())
            mape_skews.append(d["mape_skew"].item())
        ds_result = [d["ds_mse_test"].item(), d["ds_mse_mean"].item(),d["ds_mse_skew"].item()]
        ds_mse_tests.append(ds_result[0])
        ds_mse_means.append(ds_result[1])
        ds_mse_skews.append(ds_result[2])
        best = np.argmin(ds_result)
        ds_results.append(best)
        if "mape_test" in d:
            mape_result = [d["mape_test"].item(), d["mape_mean"].item(),d["mape_skew"].item()]
            mape_best = np.argmin(mape_result)
            mape_results.append(mape_best)

    print_stats("TEST", mse_tests, ds_mse_tests, mape_tests)
    print_stats("MEAN", mse_means, ds_mse_means, mape_means)
    print_stats("SKEW", mse_skews, ds_mse_skews, mape_skews)
    
    
    u,c = np.unique(ds_results, return_counts=True)
    pct_best = {0:0.0,1:0.0,2:0.0}
    for i in range(len(u)):
        pct_best[u[i]] = c[i]/len(ds_results)

    if "mape_test" in d:    
        u,c = np.unique(mape_results, return_counts=True)
        pct_mape_best = {0:0.0,1:0.0,2:0.0}
        for i in range(len(u)):
            pct_mape_best[u[i]] = c[i]/len(mape_results)

    train_mses = np.array(train_mses)
    mse_tests = np.array(mse_tests)
    mse_means = np.array(mse_means)
    mse_skews = np.array(mse_skews)

    mean_margins = mse_tests-mse_means
    skew_margins = mse_tests-mse_skews

    
    print("Pct runs where domain shift is detected: {}".format(ds_detected/len(ds_results)))
    print("Pct false positive: {}".format(false_pos/len(ds_results)))
    
    print("Pct runs where orig pred is best: {}".format(pct_best[0]))
    print("Pct runs where mean pred is best: {}".format(pct_best[1]))
    print("Pct runs where skew pred is best: {}".format(pct_best[2]))

    mean_best_idx = np.argmax(mean_margins)
    mean_worst_idx = np.argmin(mean_margins)    
    print("Best mean margin:{}, {} ({} pct)".format(fs[mean_best_idx], mean_margins[mean_best_idx], 100.0*mean_margins[mean_best_idx]/mse_tests[mean_best_idx]))
    print("Worst mean margin:{}, {} ({} pct)".format(fs[mean_worst_idx],mean_margins[mean_worst_idx],100.0*mean_margins[mean_worst_idx]/mse_tests[mean_worst_idx]))

    skew_best_idx = np.argmax(skew_margins)
    skew_worst_idx = np.argmin(skew_margins)    
    print("Best skew margin:{}, {} ({} pct)".format(fs[skew_best_idx], skew_margins[skew_best_idx], 100.0*skew_margins[skew_best_idx]/mse_tests[skew_best_idx]))
    print("Worst skew margin:{}, {} ({} pct)".format(fs[skew_worst_idx],skew_margins[skew_worst_idx],100.0*skew_margins[skew_worst_idx]/mse_tests[skew_worst_idx]))


    
    if "mape_test" in d:
        print("Pct runs where orig pred is best (mape): {}".format(pct_mape_best[0]))
        print("Pct runs where mean pred is best (mape): {}".format(pct_mape_best[1]))
        print("Pct runs where skew pred is best (mape): {}".format(pct_mape_best[2]))
    
    plot_mses(train_mses, mse_means, "train", "mean", newdir)
    plot_mses(train_mses, mse_skews, "train", "skew", newdir)
    plot_mses(mse_tests, mse_means, "test", "mean", newdir)
    plot_mses(mse_tests, mse_skews, "test", "skew", newdir)
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', help='path to output dir')
    args = parser.parse_args()
    print_results(args.outdir)
