# FuSNet + Kalman Filter

This repository runs FuSNet baseline inference and then adapts the FuSNet ReTM filters with a Kalman filter, sample by sample, for moving-source microphone recordings.

The current entry points are:

```text
run_mic13.py    13-mic setup: Group A = 5 target mics, Group B = 8 input mics
run_mic16.py    16-mic setup: Group A = 7 target mics, Group B = 9 input mics
```

The Kalman implementation is in:

```text
retm_kalman/kalman_filter.py
```

The main class is:

```python
ReTMKalmanFilterFromFuSNet
```

## Repository Layout

```text
.
├── run_mic13.py
├── run_mic16.py
├── plot.py
├── calc_covsim.py
└── retm_kalman/
    ├── kalman_filter.py
    ├── fusnet_inference_13.py
    ├── fusnet_inference_16.py
    ├── fusnet_model_13.py
    ├── fusnet_model_16.py
    └── io_utils.py
```

## Requirements

Install Python packages with CUDA-enabled PyTorch if you want GPU acceleration:

```bash
pip install numpy torch torchaudio matplotlib
```

The code automatically uses CUDA when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Input Data

Each sequence directory must contain one WAV file per microphone:

```text
mic_1.wav
mic_2.wav
mic_3.wav
...
```

For `run_mic13.py`:

```text
Group A target mics: mic_1.wav to mic_5.wav
Group B input mics : mic_6.wav to mic_13.wav
```

For `run_mic16.py`:

```text
Group A target mics: mic_1.wav to mic_7.wav
Group B input mics : mic_8.wav to mic_16.wav
```

All WAV files are loaded with `torchaudio`, converted to mono if needed, resampled to 16 kHz, trimmed to the same length, and normalized together.

## Before Running

Open the runner you want to use and edit these paths near the top of `main()`.

For 13 microphones:

```python
seq_dir = Path("/path/to/sequence/folder")
checkpoint_path = Path("/path/to/best_checkpoint_A1_1_FUSENet_13_M.pth")
out_dir = Path("results_fusnet_retm_kalman_M_1")
```

For 16 microphones:

```python
seq_dir = Path("/path/to/sequence/folder")
checkpoint_path = Path("/path/to/best_checkpoint_A1_1_FUSENet_16_P.pth")
out_dir = Path("results_fusnet_retm_kalman_P_1_16")
```

Also check that the microphone groups match your dataset:

```python
qa_mics = [...]
qb_mics = [...]
```

## Running

Run the 13-mic version:

```bash
python run_mic13.py
```

Run the 16-mic version:

```bash
python run_mic16.py
```

The script performs these steps:

1. Loads the Group A and Group B WAV files.
2. Loads the trained FuSNet checkpoint.
3. Runs baseline FuSNet inference.
4. Initializes the Kalman filter from the FuSNet convolution weights.
5. Runs sample-wise Kalman updates.
6. Saves NumPy arrays, WAV files, metrics, and the adapted model checkpoint.

<!-- ## Important Settings

The main signal settings are:

```python
context = 4096
filter_length = 2 * context + 1
window_size = 16384
stride = 8192
batch_size_for_baseline = 8
```

The main Kalman settings are:

```python
block_length = 64
transition = 0.995
process_noise = 1e-7
observation_noise = 1e-2
initial_covariance = 1e-3
adaptive_noise = True
adaptive_alpha = 0.999
adaptive_noise_floor = 1e-4
adaptive_noise_ceil = 1.0
innovation_momentum = 0.3
update_stride = 1
```

Common tuning changes:

```text
Increase process_noise if the filter adapts too slowly.
Decrease transition if old filter estimates should be forgotten faster.
Increase observation_noise if Kalman updates are too aggressive.
Increase update_stride to update less often and run faster.
``` -->

## Outputs

Each run creates files in `out_dir`:

```text
mA_target.npy
mB_input.npy
mA_fusnet_baseline.npy
mA_kalman.npy
error_kalman.npy
error_trace_rms.npy
R_adapted.npy
metrics.json
adapted_fusnet_kalman.pth
target_mic_*.wav
baseline_mic_*.wav
kalman_mic_*.wav
```

`metrics.json` contains SDR and MSE values for the baseline FuSNet output and the FuSNet + Kalman output.

## Plotting Results

Edit `RESULT_DIR` in `plot.py` so it points to the output directory you want to inspect:

```python
RESULT_DIR = Path("results_fusnet_retm_kalman_P_1_16")
```

Then run:

```bash
python plot.py
```

Plots are saved in:

```text
<RESULT_DIR>/plots_target_kalman/
```

## Covariance Similarity

Edit `RESULT_DIR` in `calc_covsim.py`:

```python
RESULT_DIR = Path("results_fusnet_retm_kalman_P_1_16")
```

Then run:

```bash
python calc_covsim.py
```

The script compares covariance similarity for the FuSNet baseline and the Kalman output when the expected `.npy` files are available.

## Notes

The current runners use hardcoded paths instead of command-line arguments. If you move datasets or checkpoints, update `seq_dir`, `checkpoint_path`, and `out_dir` before running.

Existing result folders are experiment outputs and are not required for a new run.
