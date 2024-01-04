from .utils import *
import mne_connectivity
import matplotlib.pyplot as plt
import scipy.io as sio
import os


# Define function for plotting con matrices
def plot_con_matrix(con_data, n_con_methods, connectivity_methods, roi_names, foi):
    """Visualize the connectivity matrix."""
    fig, ax = plt.subplots(1, n_con_methods, figsize=(6 * n_con_methods, 6))
    for c in range(n_con_methods):
        # Plot with imshow
        con_plot = ax[c].imshow(con_data[c, :, :, foi], cmap="binary", vmin=0, vmax=1)
        # Set title
        ax[c].set_title(connectivity_methods[c])
        # Add colorbar
        fig.colorbar(con_plot, ax=ax[c], shrink=0.7, label="Connectivity")
        # Fix labels
        ax[c].set_xticks(range(len(roi_names)))
        ax[c].set_xticklabels(roi_names)
        ax[c].set_yticks(range(len(roi_names)))
        ax[c].set_yticklabels(roi_names)
        print(
            f"Connectivity method: {connectivity_methods[c]}\n"
            + f"{con_data[c,:,:,foi]}"
        )
    return fig

# ADJUSTED BY GPT-4
def plot_connectivity(con_epochs, t_con_max, roi_names, con_methods):
    for c in range(len(con_epochs)):
        # Plot the connectivity matrix at the timepoint with highest global wPLI
        con_epochs_matrix = con_epochs[c].get_data(output="dense")[:, :, 0, t_con_max]

        # Increase figure size for better label spacing
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(con_epochs_matrix)
        fig.colorbar(im, ax=ax, label="Connectivity")

        ax.set_ylabel("Regions")
        ax.set_yticks(range(len(roi_names)))
        ax.set_yticklabels(roi_names)

        ax.set_xlabel("Regions")
        ax.set_xticks(range(len(roi_names)))
        # Adjust rotation and alignment of x labels
        ax.set_xticklabels(roi_names, rotation=60, ha='right')

        ax.set_title(f"{con_methods[c]} Connectivity")

        # Use tight layout to optimize spacing
        plt.tight_layout()
        plt.show()


def plot_connectivity_circle(con_epochs_array, con_methods, Freq_Bands, roi_names):
    import matplotlib.pyplot as plt
    import mne

    for band, method in product(Freq_Bands, con_methods):
        con = con_epochs_array[con_methods.index(method)][..., band]
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


def plot_global_connectivity(epochs, tmin, tmax, n_connections, con_epochs, Freq_Bands):
    """
    Plot the global connectivity over time.

    Args:
        epochs (Epochs): The epochs data.
        tmin (float): The minimum time value to include in the plot.
        tmax (float): The maximum time value to include in the plot.
        n_connections (int): The number of connections.
        con_epochs (list): The connectivity epochs.
        Freq_Bands (dict): The frequency bands.
    """

    # Get the timepoints within the specified time range
    times = epochs.times[(epochs.times >= tmin) & (epochs.times <= tmax)]

    for c in range(len(con_epochs)):
        # Get global average connectivity over all connections
        con_epochs_raveled_array = con_epochs[c].get_data(output="raveled")
        global_con_epochs = np.sum(con_epochs_raveled_array, axis=0) / n_connections

        for i, (k, v) in enumerate(Freq_Bands.items()):
            global_con_epochs_tmp = global_con_epochs[i]

            # Get the timepoint with the highest global connectivity right after stimulus
            t_con_max = np.argmax(global_con_epochs_tmp[times <= tmax])

            # Plot the global connectivity
            fig = plt.figure()
            plt.plot(times, global_con_epochs_tmp)
            plt.xlabel("Time (s)")
            plt.ylabel(f"Global {k} wPLI over trials")
            plt.title(f"Global {k} wPLI peaks {times[t_con_max]:.3f}s after stimulus")


def compute_connectivity_epochs(data_path, roi_names, con_methods, Freq_Bands, sfreq):
    # Load the data
    fname = os.path.join(data_path + ".stc")
    stc = mne.read_source_estimate(fname)

    # Create epochs from the data
    epochs_data = np.transpose(np.array(data), (1, 0, 2))
    _, n_channels, n_times = epochs_data.shape

    info = mne.create_info(roi_names, sfreq)
    epochs = mne.EpochsArray(epochs_data, info)

    n_freq_bands = len(Freq_Bands)
    min_freq = np.min(list(Freq_Bands.values()))
    max_freq = np.max(list(Freq_Bands.values()))

    # Prepare the freq points
    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

    fmin = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
    fmax = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])

    # We specify the connectivity measurements
    connectivity_methods = [
        "wpli",
        "dpli",
        "plv",
    ]
    n_con_methods = len(connectivity_methods)

    # Compute connectivity over trials
    con_epochs = spectral_connectivity_epochs(
        epochs,
        method=connectivity_methods,
        sfreq=sfreq,
        mode="cwt_morlet",
        cwt_freqs=freqs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        tmin=tmin,
        tmax=tmax,
        cwt_n_cycles=4,
    )

    return con_epochs


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
