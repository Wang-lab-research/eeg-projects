import utils
import preprocess
import mne_connectivity
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import numpy as np
import mne


# TODO: Load in {sub_id}_stim_labels.mat and {sub_id}_pain_ratings.mat} from processed_data_path
def separate_epochs_by_stim(sub_id, processed_data_path, stc_array, pain_thresh=None):
    # Load in stimulus labels and pain ratings
    print(f"Reading stimulus labels and pain ratings for Subject {sub_id}...")

    stim_labels = sio.loadmat(
        os.path.join(processed_data_path, sub_id + "_stim_labels.mat")
    )
    stim_labels = stim_labels["stim_labels"].tolist()[0]
    print(f"\n*stim_labels length = {len(stim_labels)}*")

    # Load in pain rating for each stimuli
    pain_ratings_raw = sio.loadmat(
        os.path.join(processed_data_path, sub_id + "_pain_ratings.mat")
    )
    pain_ratings_raw = pain_ratings_raw["pain_ratings"].tolist()[0]

    print(f"*pain_ratings_raw length = {len(pain_ratings_raw)}*\n")

    if pain_thresh is not None:
        pain_ratings = preprocess.get_binary_pain_trials(
            sub_id, pain_ratings_raw, pain_thresh, processed_data_path
        )
        # use pain/no-pain dict for counting trial ratio
        # pain_events_dict = {
        #     "Pain": 1,
        #     "No Pain": 0,
        # }
        # pain_conditions = ["Pain", "No Pain"]

    else:
        pain_ratings = pain_ratings_raw

    ##############################################################################################
    # Identify trial indices
    back_trials = [i for i, el in enumerate(stim_labels) if el > 5]
    hand_trials = [i for i, el in enumerate(stim_labels) if el <= 5]

    # delete other trials from all relevant objects
    stim_labels_hand = [el for i, el in enumerate(stim_labels) if i not in back_trials]
    stim_labels_back = [el for i, el in enumerate(stim_labels) if i not in hand_trials]
    pain_ratings_hand = [
        el for i, el in enumerate(pain_ratings) if i not in back_trials
    ]
    pain_ratings_back = [
        el for i, el in enumerate(pain_ratings) if i not in hand_trials
    ]

    deserialized_object = utils.unpickle_data(data_path)

    hand_All_Stim_epochs = (hand_LS, hand_NS, hand_HS)
    back_All_Stim_epochs = (back_LS, back_NS, back_HS)
    return hand_All_Stim_epochs, back_All_Stim_epochs, pain_ratings, stim_labels


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


def plot_connectivity(con_epochs, t_con_max, roi_names, con_methods):
    for c in range(len(con_epochs)):
        # Plot the connectivity matrix at the timepoint with highest global wPLI
        con_epochs_matrix = con_epochs[c].get_data(output="dense")[:, :, 0, t_con_max]

        fig, ax = plt.subplots()

        im = ax.imshow(con_epochs_matrix)
        fig.colorbar(im, ax=ax, label="Connectivity")

        ax.set_ylabel("Regions")
        ax.set_yticks(range(len(roi_names)))
        ax.set_yticklabels(roi_names)

        ax.set_xlabel("Regions")
        ax.set_xticks(range(len(roi_names)))
        ax.set_xticklabels(roi_names, rotation=45)

        ax.set_title(f"{con_methods[c]} Connectivity")

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
