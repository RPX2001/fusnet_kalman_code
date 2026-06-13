"""
run_kalman_v3.py  —  runner for PartitionedKalmanReTMv3
Includes per-channel diagnostics to monitor mic-5 specifically.
"""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from retm_kalman.fusnet_inference_13 import load_fusnet13_model, predict_fusnet13_original_style
from retm_kalman.kalman_fusnet_retm_v3 import PartitionedKalmanReTMv3

SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def load_mic_group(seq_dir, mic_indices, fs_target):
    signals = []
    for mic_id in mic_indices:
        wav, fs = torchaudio.load(str(Path(seq_dir)/f"mic_{mic_id}.wav"))
        if wav.shape[0]>1: wav = wav.mean(0,keepdim=True)
        if fs!=fs_target: wav = torchaudio.functional.resample(wav,fs,fs_target)
        signals.append(wav.squeeze(0))
    L = min(x.numel() for x in signals)
    return torch.stack([x[:L] for x in signals],dim=0), fs_target

def normalize_pair(mA,mB):
    peak = torch.max(torch.abs(torch.cat([mA,mB],dim=0)))
    return (mA/peak, mB/peak, float(peak)) if peak>0 else (mA,mB,1.0)

def sdr_db(ref,est,eps=1e-12):
    T=min(ref.shape[1],est.shape[1])
    err=ref[:,:T]-est[:,:T]
    return 10*np.log10((np.sum(ref[:,:T]**2,1)+eps)/(np.sum(err**2,1)+eps))

def mse_db(ref,est,eps=1e-12):
    T=min(ref.shape[1],est.shape[1])
    return 10*np.log10(np.mean((ref[:,:T]-est[:,:T])**2,1)+eps)

def print_metrics(ref,est,label):
    sdr=sdr_db(ref,est); mse=mse_db(ref,est)
    print(f"\n{'─'*72}\n  {label}\n{'─'*72}")
    print(f"  SDR per ch : {np.round(sdr,3).tolist()}")
    print(f"  SDR avg    : {np.mean(sdr):.4f} dB")
    print(f"  MSE per ch : {np.round(mse,3).tolist()}")
    print(f"  MSE avg    : {np.mean(mse):.4f} dB")

@torch.no_grad()
def run_kalman(kalman, mB, mA, context, filter_length,
               update_stride=1, log_every=8000):
    dev,dtype = kalman.device, kalman.dtype
    mB=mB.to(dev,dtype); mA=mA.to(dev,dtype)
    T=min(mB.shape[1],mA.shape[1])
    mB,mA=mB[:,:T],mA[:,:T]
    mB_pad = F.pad(mB,(context,context))
    y_out   = torch.zeros((kalman.qa,T),device=dev,dtype=dtype)
    err_out = torch.zeros((kalman.qa,T),device=dev,dtype=dtype)
    rms_log = []
    t0=time.perf_counter()
    for t in range(T):
        s=context+t
        x_full=mB_pad[:,s:s+filter_length]
        if x_full.shape[1]<filter_length:
            x_full=F.pad(x_full,(0,filter_length-x_full.shape[1]))
        d=mA[:,t]
        if update_stride<=1 or t%update_stride==0:
            y_hat,e = kalman.update_one(x_full,d)
        else:
            y_hat=kalman.predict_one(x_full); e=d-y_hat
        y_out[:,t]=y_hat; err_out[:,t]=e
        if t%log_every==0 or t==T-1:
            rms=float(torch.sqrt(torch.mean(e**2)).item())
            rms_log.append(rms)
            # per-channel running SDR
            ch_sdr=[
                f"{10*np.log10((float((mA[c,:t+1]**2).mean())+1e-12)/(float((err_out[c,:t+1]**2).mean())+1e-12)):.1f}"
                for c in range(kalman.qa)
            ]
            null_active = kalman._input_power_ema is not None
            null_str = ""
            if null_active:
                pwr = kalman._input_power_ema
                thr = kalman.null_thresh * pwr
                null_str = f"  pwr={pwr:.2e} thr={thr:.2e}"
            print(
                f"  t={t+1:07d}/{T}"
                f"  rms={rms:.3e}"
                f"  ch_SDR={ch_sdr}"
                f"{null_str}"
                f"  {time.perf_counter()-t0:.0f}s"
            )
    return (y_out.cpu().float().numpy(),
            err_out.cpu().float().numpy(),
            np.array(rms_log,np.float32))

def main():
    seq_dir = Path("/home/jaliya/eeg_speech/Julian/RetM_Workspace/Dataset/A2_rctd")
    ckpt    = Path("/home/jaliya/eeg_speech/Julian/RetM_Workspace/ReTM_Research_Project/best_checkpoint_A1_1_FUSENet_13_rctd.pth")
    out_dir = Path("results_kalman_v3"); out_dir.mkdir(parents=True, exist_ok=True)

    fs_target=16000; qa_mics=[1,2,3,4,5]; qb_mics=[6,7,8,9,10,11,12,13]
    context=4096; filter_length=2*context+1

    # ── Hyperparameters ───────────────────────────────────────────────
    # WHAT CHANGED FROM v2 AND WHY (for mic 5 specifically)
    #
    # norm_margin = 3.0
    #   Prevents the filter taps from shrinking below 1/3× their initial
    #   scale. This directly fixes the 3-4× amplitude underestimation.
    #   Lower → tighter constraint; 3.0 allows enough flexibility
    #   for the arc while bounding the collapse.
    #
    # gain_floor = 1e-4
    #   Ensures mic 5 keeps learning even when Rv is large during the
    #   null passage. Without this, K→0 and the filter freezes.
    #
    # null_power_threshold = 0.05
    #   Pauses the update when input power drops to 5% of running mean.
    #   This prevents divergence at the geometric null without needing
    #   to increase Rv aggressively.
    #
    # adaptive_noise_floor = 1e-3  (same as v2)
    #   Tighter floor than v1 — lets the filter trust good observations.
    # ─────────────────────────────────────────────────────────────────
    cfg = dict(
        block_length         = 64,
        transition           = 0.995,
        process_noise        = 1e-7,
        observation_noise    = 1e-2,
        initial_covariance   = 1e-3,
        adaptive_noise       = True,
        adaptive_alpha       = 0.999,
        adaptive_noise_floor = 1e-3,
        adaptive_noise_ceil  = 1.0,
        innovation_momentum  = 0.3,
        channel_gain_alpha   = 0.999,
        p_floor              = 1e-10,
        norm_margin          = 3.0,    # NEW: amplitude collapse prevention
        gain_floor           = 1e-4,   # NEW: keeps mic 5 updating
        null_power_threshold = 0.05,   # NEW: null-passage hold
        null_ema_alpha       = 0.999,
    )

    print("="*80)
    print("Kalman ReTM v3 — mic-5 amplitude + null-passage fixes")
    print("="*80)
    for k,v in cfg.items(): print(f"  {k:28s}: {v}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mB_cpu,_ = load_mic_group(seq_dir, qb_mics, fs_target)
    mA_cpu,_ = load_mic_group(seq_dir, qa_mics, fs_target)
    T0=min(mA_cpu.shape[1],mB_cpu.shape[1])
    mA_cpu,mB_cpu=mA_cpu[:,:T0],mB_cpu[:,:T0]
    mA_cpu,mB_cpu,scale=normalize_pair(mA_cpu,mB_cpu)

    model,_ = load_fusnet13_model(ckpt, context=context, device=str(device))
    model.eval()
    mA_fusnet = predict_fusnet13_original_style(
        model=model, mB=mB_cpu.numpy(),
        context=context, window_size=16384, stride=8192,
        batch_size=8, device=device,
    )
    T=min(mA_cpu.shape[1],mB_cpu.shape[1],mA_fusnet.shape[1])
    mA_cpu=mA_cpu[:,:T]; mB_cpu=mB_cpu[:,:T]
    mA_fusnet=mA_fusnet[:,:T]; mA_np=mA_cpu.numpy()
    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet")

    kalman = PartitionedKalmanReTMv3(
        model=model, qa=5, qb=8,
        filter_length=filter_length, device=device, dtype=torch.float32, **cfg
    )

    mA_hat,err,rms_log = run_kalman(
        kalman, mB_cpu, mA_cpu, context, filter_length
    )

    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet")
    print_metrics(mA_np, mA_hat,    "Kalman v3 (mic-5 stabilised)")

    # Per-channel table with improvement
    sdr_base=sdr_db(mA_np,mA_fusnet); sdr_v3=sdr_db(mA_np,mA_hat)
    mse_base=mse_db(mA_np,mA_fusnet); mse_v3=mse_db(mA_np,mA_hat)
    print(f"\n  {'Ch':>3} {'SDR_base':>10} {'SDR_v3':>10} {'ΔSDR':>8} {'MSE_base':>10} {'MSE_v3':>10} {'ΔMSE':>8}")
    for c in range(5):
        print(
            f"  {c+1:>3}"
            f"  {sdr_base[c]:>9.2f}"
            f"  {sdr_v3[c]:>9.2f}"
            f"  {sdr_v3[c]-sdr_base[c]:>+7.2f}"
            f"  {mse_base[c]:>9.2f}"
            f"  {mse_v3[c]:>9.2f}"
            f"  {mse_v3[c]-mse_base[c]:>+7.2f}"
        )

    kalman.copy_to_fusnet()
    np.save(out_dir/"mA_target.npy",      mA_np)
    np.save(out_dir/"mA_baseline.npy",    mA_fusnet)
    np.save(out_dir/"mA_kalman_v3.npy",   mA_hat)
    np.save(out_dir/"error_v3.npy",       err)
    np.save(out_dir/"rms_log.npy",        rms_log)
    metrics = {
        "baseline_sdr_avg": float(np.mean(sdr_base)),
        "v3_sdr_avg":       float(np.mean(sdr_v3)),
        "baseline_mse_avg": float(np.mean(mse_base)),
        "v3_mse_avg":       float(np.mean(mse_v3)),
        "per_ch": {f"ch{c+1}": {"sdr_base": float(sdr_base[c]),
                                  "sdr_v3":   float(sdr_v3[c]),
                                  "mse_base": float(mse_base[c]),
                                  "mse_v3":   float(mse_v3[c])}
                   for c in range(5)},
        "config": cfg,
    }
    with open(out_dir/"metrics.json","w") as f: json.dump(metrics,f,indent=2)
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics},
               out_dir/"adapted_fusnet_v3.pth")
    for c in range(5):
        for tag,arr in [("target",mA_np),("baseline",mA_fusnet),("v3",mA_hat)]:
            torchaudio.save(str(out_dir/f"{tag}_ch{c+1}.wav"),
                            torch.from_numpy(arr[c:c+1]).float(), fs_target)
    print(f"\nDone → {out_dir}")

if __name__=="__main__": main()