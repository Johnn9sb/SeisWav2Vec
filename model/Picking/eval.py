import os
import sys
import torch
import argparse
# =========================================================================================================
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from model import Wav2vec_Pick
from utils import parse_arguments,get_dataset
import logging
logging.getLogger().setLevel(logging.WARNING)
# =========================================================================================================
# Parameter init
args = parse_arguments()
threshold = args.threshold
model_name = args.model_name
print(model_name)
ptime = 500
window = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parl = 'y'  # y,n
# =========================================================================================================
mod_path = "/mnt/nas3/johnn9/checkpoint/"
test_name = 'threshold=' + str(threshold) + '_eval' + args.dataset
model_path = mod_path + model_name
threshold_path = model_path + '/' + test_name + '.txt'
loadmodel = model_path + '/' + 'best_checkpoint.pt' 
image_path = model_path + '/' + test_name + '_fig'
if not os.path.isdir(image_path):
    os.mkdir(image_path)
tp_path = image_path + '/tp'
fp_path = image_path + '/fp'
fpn_path = image_path + '/fpn'
tn_path = image_path + '/tn'
fn_path = image_path + '/fn'
stp_path = image_path + '/stp'
sfp_path = image_path + '/sfp'
sfpn_path = image_path + '/sfpn'
stn_path = image_path + '/stn'
sfn_path = image_path + '/sfn'
dtp_path = image_path + '/dtp'
dfp_path = image_path + '/dfp'
dtn_path = image_path + '/dtn'
dfn_path = image_path + '/dfn'
print("Init Complete!!!")
# =========================================================================================================
# DataLoad
start_time = time.time()
_,_,test = get_dataset(args)
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Load data time: {elapsed_time} sec")
print("=====================================================")
print("Data loading complete!!!")
# =========================================================================================================

def label_gen(label):
    # (B,3,3000)
    label = label[:,0,:]
    label = torch.unsqueeze(label,1)
    # other = torch.ones_like(label)-label
    # label = torch.cat((label,other), dim=1)

    return label


def image_save(batch,x,y,savepath,num,batch_num):
    if args.image == 'y':
        waveform1 = batch['X'][batch_num,0]
        waveform2 = batch['X'][batch_num,1]
        waveform3 = batch['X'][batch_num,2]
        p_predict = x[batch_num,0] 
        p_label = y[batch_num,0]
        waveform1 = waveform1.detach().numpy()
        waveform2 = waveform2.detach().numpy()
        waveform3 = waveform3.detach().numpy()
        p_predict = p_predict.detach().cpu().numpy()
        p_label = p_label.detach().cpu().numpy()
        # 绘制波形数据
        plt.figure(figsize=(10, 15))
        # 绘制波形数据
        plt.subplot(511)  # 第一行的第一个子图
        plt.plot(waveform1)
        plt.title('Waveform 1')
        plt.subplot(512)  # 第一行的第二个子图
        plt.plot(waveform2)
        plt.title('Waveform 2')
        plt.subplot(513)  # 第一行的第三个子图
        plt.plot(waveform3)
        plt.title('Waveform 3')
        plt.subplot(514)  # 第一行的第四个子图
        plt.plot(p_predict)
        plt.title('P_predict')
        plt.subplot(515) 
        plt.plot(p_label)
        plt.title('P_label')
        plt.tight_layout()
        fignum1 = str(image)
        savepath = savepath + str(num) + '.png'
        plt.savefig(savepath)
        plt.close('all')

print("Function load Complete!!!")
# =========================================================================================================
# Init
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}
p_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
}
s_dict = {
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",   
}
if args.task == 'pick':
    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
        sbg.RandomWindow(windowlen=window, strategy="pad",low=250,high=5750),
        # sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
    ]
elif args.task == 'detect':
    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=1000, windowlen=6000, selection="first", strategy="pad"),
        sbg.RandomWindow(windowlen=6000, strategy="pad",low=750,high=5000),
        # sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
        sbg.ChangeDtype(np.float32),
        sbg.DetectionLabeller(p_phases=p_dict, s_phases=s_dict),
    ]
test_gene = sbg.GenericGenerator(test)
test_gene.add_augmentations(augmentations)
test_loader = DataLoader(test_gene,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True)
print("Dataloader Complete!!!")
# =========================================================================================================
# Whole model build
if args.train_model == "wav2vec2":
    model = Wav2vec_Pick(
        device=device,
        decoder_type=args.decoder_type,
        checkpoint_path=args.checkpoint_path,
        args=args,
    )
elif args.train_model == "phasenet":
    model = sbm.PhaseNet(phases="PSN", norm="peak")
elif args.train_model == "eqt":
    if args.task == 'pick':
        model = sbm.EQTransformer(in_samples=3000, phases='PS')
    elif args.task == 'detect':
        model = sbm.EQTransformer(in_samples=6000, phases='PS')
if parl == 'y':
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_indices = list(range(num_gpus))
    model = DataParallel(model, device_ids=gpu_indices)
model.load_state_dict(torch.load(loadmodel))
model.to(device)
model.cuda(); 
model.eval()
print("Model Complete!!!")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
w2v_params = sum(p.numel() for name, p in model.named_parameters() if 'w2v' in name)
print(f"W2v params: {w2v_params}")
decoder_params = sum(p.numel() for name, p in model.named_parameters() if 'w2v' not in name)
print(f"Decoder params: {decoder_params}")
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Model Complete time: {elapsed_time} sec")
print("=====================================================")
# =========================================================================================================
# Testing
print("Testing start!!!")
start_time = time.time()
f = open(threshold_path,"w")

result_path = '/mnt/nas3/johnn9/experiment/pickmag/pick_result'
result_num = 0 
if args.task == 'pick':   
    print("Testing: " + str(threshold) + " start!!")
    p_tp,p_tn,p_fp,p_fn,p_fpn = 0,0,0,0,0
    s_tp,s_tn,s_fp,s_fn,s_fpn = 0,0,0,0,0
    p_mean,p_std,p_mae = 0,0,0
    s_mean,s_std,s_mae = 0,0,0
    image = 0
    # Test loop
    progre = tqdm(test_loader,total = len(test_loader), ncols=80)

    for batch in progre:
        p_mean_batch,p_std_batch,p_mae_batch = 0,0,0
        s_mean_batch,s_std_batch,s_mae_batch = 0,0,0
        x = batch['X'].to(device)
        y = batch['y'].to(device)
        x = model(x.to(device))
        
        file_name = f"batch_{result_num}.pt"
        file_path = os.path.join(result_path, file_name)
        torch.save(x, file_path)
        result_num = result_num + 1

        if args.train_model == 'eqt':
            x_tensor = torch.empty(2,len(y),window)
            for index, item in enumerate(x):
                x_tensor[index] = item
                if index == 1:
                    break
            x = x_tensor.permute(1,0,2)
            x = x.to(device)
        for num in range(len(x)):
            xp = x[num,0]
            xs = x[num,1]
            yp = y[num,0]
            ys = y[num,1]
                
            if torch.max(yp) >= threshold and torch.max(xp) >= threshold:
                if abs(torch.argmax(yp).item() - torch.argmax(xp).item()) <= 50:
                    p_tp = p_tp + 1
                    if p_tp < 10:
                        image_save(batch,x,y,tp_path,p_tp,num)
                else:
                    p_fp = p_fp + 1
                    if p_fp < 10:
                        image_save(batch,x,y,fp_path,p_fp,num)
            if torch.max(yp) < threshold and torch.max(xp) >= threshold:
                p_fpn = p_fpn + 1
                if p_fpn < 10:
                    image_save(batch,x,y,fpn_path,p_fpn,num)
            if torch.max(yp) >= threshold and torch.max(xp) < threshold:
                p_fn = p_fn + 1
                if p_fn < 10:
                    image_save(batch,x,y,fn_path,p_fn,num)
            if torch.max(yp) < threshold and torch.max(xp) < threshold:
                p_tn = p_tn + 1
                if p_tn < 10:
                    image_save(batch,x,y,tn_path,p_tn,num)

            if torch.max(ys) >= threshold and torch.max(xs) >= threshold:
                if abs(torch.argmax(ys).item() - torch.argmax(xs).item()) <= 50:
                    s_tp = s_tp + 1
                    if s_tp < 10:
                        image_save(batch,x,y,stp_path,s_tp,num)
                else:
                    s_fp = s_fp + 1
                    if s_fp < 10:
                        image_save(batch,x,y,sfp_path,s_fp,num)
            if torch.max(ys) < threshold and torch.max(xs) >= threshold:
                s_fpn = s_fpn + 1
                if s_fpn < 10:
                    image_save(batch,x,y,sfpn_path,s_fpn,num)
            if torch.max(ys) >= threshold and torch.max(xs) < threshold:
                s_fn = s_fn + 1
                if s_fn < 10:
                    image_save(batch,x,y,sfn_path,s_fn,num)
            if torch.max(ys) < threshold and torch.max(xs) < threshold:
                s_tn = s_tn + 1
                if s_tn < 10:
                    image_save(batch,x,y,stn_path,s_tn,num)

            p_mean_now = torch.mean(xp - yp)
            p_mean_batch = p_mean_batch + p_mean_now.item()
            p_std_now = torch.std(xp - yp)
            p_std_batch = p_std_batch + p_std_now.item()
            p_mae_now = torch.mean(torch.abs(xp - yp))
            p_mae_batch = p_mae_batch + p_mae_now.item()
            s_mean_now = torch.mean(xs - ys)
            s_mean_batch = s_mean_batch + s_mean_now.item()
            s_std_now = torch.std(xs - ys)
            s_std_batch = s_std_batch + s_std_now.item()
            s_mae_now = torch.mean(torch.abs(xs - ys))
            s_mae_batch = s_mae_batch + s_mae_now.item()   
        p_mean = p_mean + (p_mean_batch / args.batch_size)
        p_std = p_std + (p_std_batch / args.batch_size)
        p_mae = p_mae + (p_mae_batch / args.batch_size)
        s_mean = s_mean + (s_mean_batch / args.batch_size)
        s_std = s_std + (s_std_batch / args.batch_size)
        s_mae = s_mae + (s_mae_batch / args.batch_size)
        progre.set_postfix({"TP": p_tp, "FP": p_fp+p_fpn, "TN": p_tn, "FN": p_fn})

        if args.test_mode == 'true':
            break
    p_mean = p_mean / len(test_loader)
    p_std = p_std / len(test_loader)
    p_mae = p_mae / len(test_loader)
    s_mean = s_mean / len(test_loader)
    s_std = s_std / len(test_loader)
    s_mae = s_mae / len(test_loader)

    # 計算分數 
    p_fp = p_fp + p_fpn
    s_fp = s_fp + s_fpn

    if p_tp == 0:
        p_recall = 0
        p_precision = 0
        p_f1 = 0
    else:
        p_recall = p_tp / (p_tp + p_fn)
        p_precision = p_tp / (p_tp + p_fp)
        p_f1 = 2*((p_precision * p_recall)/(p_precision+p_recall))
    if s_tp == 0:
        s_recall = 0
        s_precision = 0
        s_f1 = 0
    else:
        s_recall = s_tp / (s_tp + s_fn)
        s_precision = s_tp / (s_tp + s_fp)
        s_f1 = 2*((s_precision * s_recall)/(s_precision+s_recall))
    # Write Log
    f.write(args.dataset + ' dataset')
    f.write(args.dataset + ' dataset')
    f.write(args.dataset + ' dataset')
    f.write('==================================================' + '\n')
    f.write('Threshold = ' + str(threshold) + '\n')
    f.write('P-phase precision = ' + str(p_precision) + '\n')
    f.write('P-phase recall = ' + str(p_recall) + '\n')
    f.write('P-phase f1score = ' + str(p_f1) + '\n')
    f.write('P-phase mean = ' + str(p_mean) + '\n')
    f.write('P-phase std = ' + str(p_std) + '\n')
    f.write('P-phase mae = ' + str(p_mae) + '\n')
    f.write('P-phase tp = ' + str(p_tp) + '\n')
    f.write('P-phase fp = ' + str(p_fp) + '\n')
    f.write('P-phase tn = ' + str(p_tn) + '\n')
    f.write('P-phase fn = ' + str(p_fn) + '\n')
    f.write('==================================================' + '\n')
    f.write('S-phase precision = ' + str(s_precision) + '\n')
    f.write('S-phase recall = ' + str(s_recall) + '\n')
    f.write('S-phase f1score = ' + str(s_f1) + '\n')
    f.write('S-phase mean = ' + str(s_mean) + '\n')
    f.write('S-phase std = ' + str(s_std) + '\n')
    f.write('S-phase mae = ' + str(s_mae) + '\n')
    f.write('S-phase tp = ' + str(s_tp) + '\n')
    f.write('S-phase fp = ' + str(s_fp) + '\n')
    f.write('S-phase tn = ' + str(s_tn) + '\n')
    f.write('S-phase fn = ' + str(s_fn) + '\n')
    
    print(args.dataset + ' dataset')
    print(args.dataset + ' dataset')
    print(args.dataset + ' dataset')
    print('==================================================')
    print('Threshold = ' + str(threshold))
    print('P-phase precision = ' + str(p_precision))
    print('P-phase recall = ' + str(p_recall))
    print('P-phase f1score = ' + str(p_f1))
    print('P-phase mean = ' + str(p_mean))
    print('P-phase std = ' + str(p_std))
    print('P-phase mae = ' + str(p_mae))
    print('P-phase tp = ' + str(p_tp))
    print('P-phase fp = ' + str(p_fp))
    print('P-phase tn = ' + str(p_tn))
    print('P-phase fn = ' + str(p_fn))
    print('==================================================')
    print('S-phase precision = ' + str(s_precision))
    print('S-phase recall = ' + str(s_recall))
    print('S-phase f1score = ' + str(s_f1))
    print('S-phase mean = ' + str(s_mean))
    print('S-phase std = ' + str(s_std))
    print('S-phase mae = ' + str(s_mae))
    print('S-phase tp = ' + str(s_tp))
    print('S-phase fp = ' + str(s_fp))
    print('S-phase tn = ' + str(s_tn))
    print('S-phase fn = ' + str(s_fn))

elif args.task == 'detect':
    print("Testing: " + str(threshold) + " start!!")
    d_tp,d_tn,d_fp,d_fn = 0,0,0,0
    d_mean,d_std,d_mae = 0,0,0
    image = 0
    # Test loop
    progre = tqdm(test_loader,total = len(test_loader), ncols=80)
    for batch in progre:
        d_mean_batch,d_std_batch,d_mae_batch = 0,0,0
        x = batch['X'].to(device)
        y = batch['y'].to(device)
        x = model(x.to(device))

        if args.train_model == 'eqt':
            x_tensor = torch.empty(1,len(y),6000)
            for index, item in enumerate(x):
                x_tensor[index] = item
                if index == 0:
                    break
            x = x_tensor.permute(1,0,2)
            x = x.to(device)
        for num in range(len(x)):
            xd = x[num,0]
            yd = y[num,0]

            if torch.max(yd) >= threshold and torch.max(xd) >= threshold:
                d_tp = d_tp + 1
                if d_tp < 10:
                    image_save(batch,x,y,dtp_path,d_tp,num)
            if torch.max(yd) < threshold and torch.max(xd) >= threshold:
                d_fp = d_fp + 1
                if d_fp < 10:
                    image_save(batch,x,y,dfp_path,d_fp,num)
            if torch.max(yd) >= threshold and torch.max(xd) < threshold:
                d_fn = d_fn + 1
                if d_fn < 10:
                    image_save(batch,x,y,dfn_path,d_fn,num)
            if torch.max(yd) < threshold and torch.max(xd) < threshold:
                d_tn = d_tn + 1
                if d_tn < 10:
                    image_save(batch,x,y,dtn_path,d_tn,num)

            d_mean_now = torch.mean(xd - yd)
            d_mean_batch = d_mean_batch + d_mean_now.item()
            d_std_now = torch.std(xd - yd)
            d_std_batch = d_std_batch + d_std_now.item()
            d_mae_now = torch.mean(torch.abs(xd - yd))
            d_mae_batch = d_mae_batch + d_mae_now.item()
        d_mean = d_mean + (d_mean_batch / args.batch_size)
        d_std = d_std + (d_std_batch / args.batch_size)
        d_mae = d_mae + (d_mae_batch / args.batch_size)
        progre.set_postfix({"TP": d_tp, "FP": d_fp, "TN": d_tn, "FN": d_fn})

        if args.test_mode == 'true':
            break
    d_mean = d_mean / len(test_loader)
    d_std = d_std / len(test_loader)
    d_mae = d_mae / len(test_loader)

    # 計算分數 

    if d_tp == 0:
        d_recall = 0
        d_precision = 0
        d_f1 = 0
    else:
        d_recall = d_tp / (d_tp + d_fn)
        d_precision = d_tp / (d_tp + d_fp)
        d_f1 = 2*((d_precision * d_recall)/(d_precision + d_recall))
    # Write Log
    f.write('==================================================' + '\n')
    f.write('Threshold = ' + str(threshold) + '\n')
    f.write('d-phase precision = ' + str(d_precision) + '\n')
    f.write('d-phase recall = ' + str(d_recall) + '\n')
    f.write('d-phase f1score = ' + str(d_f1) + '\n')
    f.write('d-phase mean = ' + str(d_mean) + '\n')
    f.write('d-phase std = ' + str(d_std) + '\n')
    f.write('d-phase mae = ' + str(d_mae) + '\n')
    f.write('d-phase tp = ' + str(d_tp) + '\n')
    f.write('d-phase fp = ' + str(d_fp) + '\n')
    f.write('d-phase tn = ' + str(d_tn) + '\n')
    f.write('d-phase fn = ' + str(d_fn) + '\n')
    
    print('==================================================')
    print('Threshold = ' + str(threshold))
    print('d-phase precision = ' + str(d_precision))
    print('d-phase recall = ' + str(d_recall))
    print('d-phase f1score = ' + str(d_f1))
    print('d-phase mean = ' + str(d_mean))
    print('d-phase std = ' + str(d_std))
    print('d-phase mae = ' + str(d_mae))
    print('d-phase tp = ' + str(d_tp))
    print('d-phase fp = ' + str(d_fp))
    print('d-phase tn = ' + str(d_tn))
    print('d-phase fn = ' + str(d_fn))

f.close()
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Testing time: {elapsed_time} sec")
print("=====================================================")
sys.exit()