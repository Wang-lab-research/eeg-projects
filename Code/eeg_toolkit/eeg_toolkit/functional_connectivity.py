from .utils import *
import mne_connectivity
import matplotlib.pyplot as plt
import scipy.io as sio
import os


def compute_connectivity_epochs(data_path, roi_names, con_methods, Freq_Bands, sfreq):
    # Load the data
    data = []
    for roi in roi_names:
        data_file = os.path.join(data_path, roi + "_stc_zepo.mat")
        tmp_data = sio.loadmat(data_file)["data"]

        data.append(tmp_data)  # Assumes .mat files are named after the ROI

    # Create epochs from the data
    epochs_data = np.transpose(np.array(data), (1, 0, 2))
    _, n_channels, n_times = epochs_data.shape

    info = mne.create_info(
        roi_names, sfreq
    )  # Assumes same sampling frequency for all data
    epochs = mne.EpochsArray(epochs_data, info)

    n_con_methods = len(con_methods)
    n_freq_bands = len(Freq_Bands)

    # Pre-allocatate memory for the connectivity matrices
    con_epochs_array = np.zeros(
        (n_con_methods, n_channels, n_channels, n_freq_bands, n_times)
    )
    con_epochs_array[con_epochs_array == 0] = np.nan  # nan matrix

    # Compute connectivity
    for band, freq in Freq_Bands.items():
        for method in con_methods:
            con = mne_connectivity.spectral_connectivity_epochs(
                epochs,
                method=method,
                mode="multitaper",
                sfreq=sfreq,
                fmin=freq[0],
                fmax=freq[1],
                faverage=True,
                mt_adaptive=False,
                n_jobs=-1,
            )

    for c in range(n_con_methods):
        con_epochs_array[c] = con_epochs[c].get_data(output="dense")

    # Plotting
    for (band, method), con in con_matrices.items():
        plt.figure(figsize=(10, 8))
        mne.viz.plot_connectivity_circle(
            con,
            roi_names,
            title=f"{band.upper()} {method.upper()} Epochs Connectivity",
            facecolor="white",
            textcolor="black",
            node_edgecolor="black",
            fontsize_names=8,
        )
        plt.show()


def compute_connectivity_resting_state(
    data_path, roi_names, con_methods, Freq_Bands, sfreq, condition
):
    # Load the data
    data = {}
    for roi in roi_names:
        data_file = os.path.join(data_path, roi + ".mat")
        data[roi] = sio.loadmat(data_file)  # Assumes .mat files are named after the ROI

    # Create raw data from the data
    raw_data = np.array([data[roi] for roi in roi_names])
    info = mne.create_info(
        roi_names, sfreq
    )  # Assumes same sampling frequency for all data
    raw = mne.io.RawArray(raw_data, info)

    # Compute connectivity
    con_matrices = {}
    for band, freq in Freq_Bands.items():
        for method in con_methods:
            (
                con,
                freqs,
                times,
                n_epochs,
                n_tapers,
            ) = mne_connectivity.spectral_connectivity(
                raw,
                method=method,
                mode="multitaper",
                sfreq=sfreq,
                fmin=freq[0],
                fmax=freq[1],
                faverage=True,
                mt_adaptive=True,
                n_jobs=1,
            )
            con_matrices[(band, method)] = con

    # Plotting
    for (band, method), con in con_matrices.items():
        plt.figure(figsize=(10, 8))
        mne.viz.plot_connectivity_circle(
            con,
            roi_names,
            title=f"{condition.upper()} Condition {band.upper()} {method.upper()} Connectivity",
            facecolor="white",
            textcolor="black",
            node_edgecolor="black",
            fontsize_names=8,
        )
        plt.show()
