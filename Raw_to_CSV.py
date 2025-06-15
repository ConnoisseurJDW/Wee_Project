import os
import mne
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.signal import coherence
from scipy.integrate import trapezoid

# PARAMETERS
data_folder = os.getcwd()
output_csv = os.path.join(data_folder, "LEMON_EEG.csv")

bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 25),
    'highbeta': (25, 30),
    'gamma': (30, 40)
}

# compute bandpower as absolute power (uV²)
def compute_bandpower(psd, freqs, fmin, fmax):
    idx_band = np.logical_and(freqs >= fmin, freqs < fmax)
    psd_uV = psd * (1e6)**2  # V²/Hz → uV²/Hz
    power = trapezoid(psd_uV[:, idx_band], freqs[idx_band])
    return power

# compute coherence (scaled *100)
def compute_coherence_epochs(epochs_data, sfreq, ch_names, band, fmin, fmax):
    n_channels = epochs_data.shape[1]
    n_epochs = epochs_data.shape[0]
    coh_features = {}

    nperseg = min(256, epochs_data.shape[2])  

    for i, j in combinations(range(n_channels), 2):
        coh_band_values = []

        for epoch_idx in range(n_epochs):
            f, Cxy = coherence(epochs_data[epoch_idx, i], epochs_data[epoch_idx, j], fs=sfreq, nperseg=nperseg)
            idx_band = np.logical_and(f >= fmin, f < fmax)

            if len(f[idx_band]) == 0:
                band_coh = np.nan
                print(f"empty band {band} for COH → {ch_names[i]}-{ch_names[j]} (epoch {epoch_idx})")
            else:
                band_coh = np.mean(Cxy[idx_band]) * 100

            coh_band_values.append(band_coh)

        # Average across epochs
        band_coh_mean = np.nanmean(coh_band_values)

        key = f"COH.{band}.{ch_names[i]}.{ch_names[j]}"
        coh_features[key] = band_coh_mean

    return coh_features

all_subject_features = []

for sub_id in os.listdir(data_folder):
    if not sub_id.startswith("sub-"):
        continue

    sub_path = os.path.join(data_folder, sub_id)
    if not os.path.isdir(sub_path):
        continue

    vhdr_file = None
    set_file = None

    for root, dirs, files in os.walk(sub_path):
        if ".ipynb_checkpoints" in root:
            continue

        for file in files:
            if file.endswith(".vhdr") and vhdr_file is None:
                vhdr_file = os.path.join(root, file)
            elif file.endswith(".set") and set_file is None:
                set_file = os.path.join(root, file)

        if vhdr_file or set_file:
            break

    if vhdr_file is None and set_file is None:
        continue

    # Load
    data = None
    try:
        if vhdr_file is not None:
            data = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        elif set_file is not None:
            data = mne.io.read_raw_eeglab(set_file, preload=True)
    except Exception as e:
        continue

    # Skip if already epoched
    if isinstance(data, mne.Epochs):
        continue

    # Set reference (optional)
    try:
        data.set_eeg_reference(ref_channels=['A1', 'A2'])
    except:
        pass

    # Filter 1-40 Hz
    data.filter(1., 40., method='iir', iir_params=dict(order=5, ftype='butter'))
    data.set_annotations(mne.Annotations([], [], []))

    # Create 2 s epochs
    epochs = mne.make_fixed_length_epochs(data, duration=2.0, preload=True)
    print(f"{sub_id}: {len(epochs)} epochs")

    # PSD
    psd = epochs.compute_psd(fmin=0.5, fmax=40, method='welch', average='mean')
    psds, freqs = psd.get_data(return_freqs=True)

    ch_names = data.info['ch_names']
    features = {"subject_ID": sub_id}

    # PSD features
    psds_mean = psds.mean(axis=0)  # average across epochs
    for band, (fmin, fmax) in bands.items():
        bp = compute_bandpower(psds_mean, freqs, fmin, fmax)
        for ch_name, val in zip(ch_names, bp):
            key = f"PSD.{band}.{ch_name}"
            features[key] = val

    # COH features
    epochs_data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    sfreq = data.info['sfreq']

    for band, (fmin, fmax) in bands.items():
        coh_band = compute_coherence_epochs(epochs_data, sfreq, ch_names, band, fmin, fmax)
        features.update(coh_band)

    # Append subject
    all_subject_features.append(features)

# SAVE
df = pd.DataFrame(all_subject_features)

# Sort columns
subject_col = ["subject_ID"]
psd_cols = sorted([col for col in df.columns if col.startswith("PSD.")])
coh_cols = sorted([col for col in df.columns if col.startswith("COH.")])
sorted_cols = subject_col + psd_cols + coh_cols

df = df[sorted_cols]

# Save CSV
df.to_csv(output_csv, index=False)
print(f"\nSaved: {output_csv}")
