from eeg_toolkit import utils, preprocess
import mne_connectivity as mne_conn
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import numpy as np
import mne


def get_info_by_stim(stim_label, stim_labels, pain_ratings):
    """
    Get the information by stimulus label.

    Parameters:
        stim_label (any): The stimulus label to search for.
        stim_labels (list): A list of stimulus labels.
        pain_ratings (list): A list of pain ratings.

    Returns:
        tuple: A tuple containing two lists. The first list contains the indices of the stimulus labels that match the given stim_label. The second list contains the corresponding pain ratings for those stimulus labels.
    """
    site_stim_labels = [i for i, el in enumerate(stim_labels) if el == stim_label]
    site_stim_ratings = [
        el for i, el in enumerate(pain_ratings) if i in site_stim_labels
    ]

    return site_stim_labels, site_stim_ratings


# Load in {sub_id}_stim_labels.mat and {sub_id}_pain_ratings.mat} from processed_data_path
def separate_epochs_by_stim(
    sub_id, processed_data_path, stc_data_path, pain_thresh=None
):
    """
    Separates epochs by stimulus for a given subject.

    :param sub_id: The ID of the subject.
    :param processed_data_path: The path to the processed data.
    :param stc_data_path: The path to the label time courses.
    :param pain_thresh: The pain threshold. Default is None.

    Returns:
    - hand_all_label_ts: A tuple containing the label time courses for hand stimuli.
    - back_all_label_ts: A tuple containing the label time courses for back stimuli.
    - hand_all_ratings: A tuple containing the pain ratings for hand stimuli.
    - back_all_ratings: A tuple containing the pain ratings for back stimuli.
    """
    ##############################################################################################

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

    hand_NS_labels, hand_NS_ratings = get_info_by_stim(5, stim_labels, pain_ratings)
    hand_LS_labels, hand_LS_ratings = get_info_by_stim(4, stim_labels, pain_ratings)
    hand_HS_labels, hand_HS_ratings = get_info_by_stim(3, stim_labels, pain_ratings)

    back_NS_labels, back_NS_ratings = get_info_by_stim(8, stim_labels, pain_ratings)
    back_LS_labels, back_LS_ratings = get_info_by_stim(7, stim_labels, pain_ratings)
    back_HS_labels, back_HS_ratings = get_info_by_stim(6, stim_labels, pain_ratings)

    ##############################################################################################
    # Load in label time courses and separate epochs by stimulus

    label_ts = utils.unpickle_data(stc_data_path / f"{sub_id}_epochs.pkl")
    hand_NS_label_ts = [el for i, el in enumerate(label_ts) if i in hand_NS_labels]
    hand_LS_label_ts = [el for i, el in enumerate(label_ts) if i in hand_LS_labels]
    hand_HS_label_ts = [el for i, el in enumerate(label_ts) if i in hand_HS_labels]

    back_NS_label_ts = [el for i, el in enumerate(label_ts) if i in back_NS_labels]
    back_LS_label_ts = [el for i, el in enumerate(label_ts) if i in back_LS_labels]
    back_HS_label_ts = [el for i, el in enumerate(label_ts) if i in back_HS_labels]

    ##############################################################################################
    # Return labels and pain ratings
    hand_all_label_ts = (hand_NS_label_ts, hand_LS_label_ts, hand_HS_label_ts)
    back_all_label_ts = (back_NS_label_ts, back_LS_label_ts, back_HS_label_ts)

    hand_all_ratings = (hand_NS_ratings, hand_LS_ratings, hand_HS_ratings)
    back_all_ratings = (back_NS_ratings, back_LS_ratings, back_HS_ratings)

    return hand_all_label_ts, back_all_label_ts, hand_all_ratings, back_all_ratings


def compute_connectivity_epochs(
    label_ts, roi_names, method, Freq_Bands, tmin, tmax, sfreq=400
):
    # Specify the frequency band
    min_freq = np.min(list(Freq_Bands.values()))
    max_freq = np.max(list(Freq_Bands.values()))

    # Prepare the freq points
    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

    fmin = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
    fmax = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])

    # Compute connectivity over trials
    con_epochs = mne_conn.spectral_connectivity_epochs(
        epochs,
        method=method,
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
    label_ts, roi_names, method, Freq_Bands, sfreq, condition
):
    con_matrices = {}
    for band, freq in Freq_Bands.items():
        (
            con,
            freqs,
            times,
            n_epochs,
            n_tapers,
        ) = mne_conn.spectral_connectivity(
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


def plot_connectivity_circle(con_matrices, con_methods, Freq_Bands, roi_names):
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
