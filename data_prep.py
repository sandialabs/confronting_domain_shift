#Data prep code based on open source repo:
#    https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py
import numpy as np
import pandas as pd
import torch

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, input_window, output_window):
    in_seq = []
    out_seq = []
    L = len(input_data)
    for i in range(L-input_window-output_window):
        train_seq = input_data[i:i+input_window]
        train_label = input_data[i+1:i+input_window+output_window]
        in_seq.append(train_seq)
        out_seq.append(train_label)
    return torch.FloatTensor(in_seq), torch.FloatTensor(out_seq)

def get_data(datapath, input_window, output_window, downsample_factor=400, n_train=3000, n_test=1000):
    data = gen_dataset(datapath, downsample_factor=downsample_factor)
    train_data = data[:n_train]
    val_data = data[n_train:-n_test]
    test_data = data[-n_test:]
    
    train_sequence, train_lbl = create_inout_sequences(train_data,input_window, output_window)
    train_sequence = train_sequence[:-output_window]
    train_lbl = train_lbl[:-output_window]
    val_sequence, val_lbl = create_inout_sequences(val_data,input_window, output_window)
    val_sequence = val_sequence[:-output_window]
    val_lbl = val_lbl[:-output_window]
    test_sequence, test_lbl = create_inout_sequences(test_data,input_window, output_window)
    test_sequence = test_sequence[:-output_window]
    test_lbl = test_lbl[:-output_window]
    
    return train_sequence,train_lbl,val_sequence,val_lbl,test_sequence,test_lbl

def get_batch(seq,lbl,i,batch_size, input_window, output_window):
    seq_len = min(batch_size, len(seq) - 1 - i)
    #print("SEQ", (seq.size()), "I", i, "LEN", seq_len)
    data = seq[i:i+seq_len]  
    lbl = lbl[i:i+seq_len]
    input = torch.stack(torch.stack([item for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item for item in lbl]).chunk(input_window+output_window,1))
    return input, target


def gen_dataset(path, downsample_factor=400):
    df = pd.read_csv(path)
    arr = np.array(df)
    data_idx = 0
    if arr.shape[1] > 1:
        data_idx = 1
    displacement = np.array(df[list(df.columns)[data_idx]].values)
    displacement = displacement[range(0,len(displacement),downsample_factor)] 
    return displacement

def find_intersection(train_data, test_data, thresh=0.05):
    closest_pt = -1
    for j in range(len(test_data - 1)):
        test_sign = test_data[j+1]-test_data[j] > 0
        diff = np.abs(test_data[j] - train_data)
        closest_pts = np.where(diff <= thresh)[0]
        
        for i in range(len(closest_pts)-1):
            lo = closest_pts[i]
            hi = closest_pts[i+1]
            if hi - lo == 1 and hi<len(train_data):
                train_sign = train_data[hi]-train_data[lo] > 0
                if test_sign and train_sign and train_data[lo] <= test_data[j] and train_data[hi] >= test_data[j]:
                    return closest_pts[i+1], j
                if (not test_sign) and (not train_sign) and train_data[lo] >= test_data[j] and train_data[hi] <= test_data[j]:
                    return closest_pts[i+1], j
    print("No intersection found!")
    return 0
            
