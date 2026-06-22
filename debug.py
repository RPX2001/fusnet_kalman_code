import numpy as np

old = np.load("old_results/mA_fusnet_initial.npy")
new = np.load("results_fusnet_parameter_kalman_gpu_13/mA_fusnet_baseline.npy")

T = min(old.shape[1], new.shape[1])
old = old[:, :T]
new = new[:, :T]

print("old shape:", old.shape)
print("new shape:", new.shape)
print("max abs diff:", np.max(np.abs(old - new)))
print("mean abs diff:", np.mean(np.abs(old - new)))

for ch in range(old.shape[0]):
    corr = np.corrcoef(old[ch], new[ch])[0, 1]
    print(f"channel {ch+1} correlation:", corr)