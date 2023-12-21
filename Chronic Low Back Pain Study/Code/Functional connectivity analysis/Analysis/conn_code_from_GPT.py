import mne
import mne_connectivity
import numpy as np
import matplotlib.pyplot as plt

# Load your data
# Note: Replace 'your_data_file.fif' with your actual EEG data file path
raw = mne.io.read_raw_fif('your_data_file.fif', preload=True)

# Assuming your data is already epoched and source localized
# If not, you need to perform these steps before proceeding

# Average the source points
# This is a simplistic approach, and you may need a more sophisticated method depending on your data structure
data = raw.get_data()
mean_data = np.mean(data, axis=1)

# Create epochs from the averaged data
epochs = mne.EpochsArray(mean_data, raw.info)

# Define frequency bands
fmin, fmax = 8, 13  # Example for alpha band

# Compute connectivity
con_methods = ['wpli', 'plv', 'aac']
con_matrices = {}

for method in con_methods:
    con, freqs, times, n_epochs, n_tapers = mne_connectivity.spectral_connectivity(
        epochs, method=method, mode='multitaper', sfreq=raw.info['sfreq'],
        fmin=fmin, fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
    con_matrices[method] = con

# Plotting
for method, con in con_matrices.items():
    plt.figure(figsize=(10, 8))
    mne.viz.plot_connectivity_circle(con.get_data(), epoch.ch_names,
                                     title=method.upper() + ' Connectivity',
                                     facecolor='white', textcolor='black', node_edgecolor='black', fontsize_names=8)
    plt.show()
