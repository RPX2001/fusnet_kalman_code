# FuSNet + Kalman Dynamic ReTM Correction System

This package integrates your FuSNet-7 model directly into the project folder and runs the full proposed system from a config file.

The implementation follows this algorithm:

```text
mB(t) → FuSNet → mA_hat_F(t)
error → Kalman ReTM correction estimator → R_delta(t)
R_delta(t), mB(t) → correction filter → ΔmA_hat(t)
mA_hat(t) = mA_hat_F(t) + ΔmA_hat(t)
e(t) = mA(t) - mA_hat(t)
```

Your FuSNet model is included in:

```text
retm_kalman/fusnet_model.py
```

It is based on the model class you provided:

```python
class MultiChannelConvolutionModel(nn.Module):
    ...
```

## Folder structure

```text
fusnet_kalman_configured/
├── config.json
├── run_system.py
├── requirements.txt
├── README.md
└── retm_kalman/
    ├── __init__.py
    ├── fusnet_model.py
    ├── fusnet_inference.py
    ├── io_utils.py
    ├── kalman_full.py
    ├── kalman_block.py
    └── metrics.py
```

## Input sequence format

Each test sequence folder must contain:

```text
seq_001/
├── mic_1.wav
├── mic_2.wav
├── mic_3.wav
├── mic_4.wav
├── mic_5.wav
├── mic_6.wav
└── mic_7.wav
```

Group A target:

```text
mic_1, mic_2, mic_3
```

Group B input:

```text
mic_4, mic_5, mic_6, mic_7
```

## How to run

First edit `config.json`:

```json
"seq_dir": "data/moving/A1/seq_001",
"out_dir": "outputs/fusnet_kalman_seq001",
"checkpoint": "checkpoints/FUSENet_7/checkpoint2_AWGN_FUSENet_7.pth"
```

Then run:

```bash
python run_system.py
```

No command-line arguments are needed.

## Full Kalman vs Block Kalman

In `config.json`, choose:

```json
"mode": "full"
```

or:

```json
"mode": "block"
```

Recommended settings:

```text
Full Kalman:  L = 128 or 256
Block Kalman: L = 512, 1024, or 2048
```

## Important FuSNet settings

Your original code used:

```python
context = 2**13 // 2       # 4096
filter_length = 2*context + 1   # 8193
window_size = 2**14        # 16384
stride = 2**13             # 8192
```

These are now stored in `config.json`.

## Outputs

The output folder contains:

```text
mA_target.npy
mB_input.npy
mA_fusnet_initial.npy
delta_kalman.npy
mA_final_kalman.npy
error_final.npy
metrics.json
used_config.json
wav_target_mA/
wav_fusnet_initial/
wav_kalman_final/
wav_kalman_delta/
```

