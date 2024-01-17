from eeg_toolkit import utils, preprocess
import mne
import mne_connectivity as mne_conn
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import numpy as np
from tabulate import tabulate

from mne.datasets import fetch_fsaverage

fs_dir = fetch_fsaverage(verbose=True)
subject = "fsaverage"
subjects_dir = os.path.dirname(fs_dir)


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
    sub_id, processed_data_path, stc_data_path, include_LS=False, pain_thresh=None
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
        os.path.join(processed_data_path, f"{sub_id}_stim_labels.mat")
    )
    stim_labels = stim_labels["stim_labels"].tolist()[0]
    print(f"\n*stim_labels length = {len(stim_labels)}*")

    # Load in pain rating for each stimuli
    pain_ratings_raw = sio.loadmat(
        os.path.join(processed_data_path, f"{sub_id}_pain_ratings.mat")
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
    hand_HS_labels, hand_HS_ratings = get_info_by_stim(3, stim_labels, pain_ratings)

    back_NS_labels, back_NS_ratings = get_info_by_stim(8, stim_labels, pain_ratings)
    back_HS_labels, back_HS_ratings = get_info_by_stim(6, stim_labels, pain_ratings)

    ##############################################################################################
    # Load in label time courses and separate epochs by stimulus

    label_ts = utils.unpickle_data(stc_data_path / f"{sub_id}_epochs.pkl")
    hand_NS_label_ts = [el for i, el in enumerate(label_ts) if i in hand_NS_labels]
    hand_HS_label_ts = [el for i, el in enumerate(label_ts) if i in hand_HS_labels]

    back_NS_label_ts = [el for i, el in enumerate(label_ts) if i in back_NS_labels]
    back_HS_label_ts = [el for i, el in enumerate(label_ts) if i in back_HS_labels]

    ##############################################################################################
    # Include LS only if specified
    if include_LS:
        hand_LS_labels, hand_LS_ratings = get_info_by_stim(4, stim_labels, pain_ratings)
        back_LS_labels, back_LS_ratings = get_info_by_stim(7, stim_labels, pain_ratings)

        hand_LS_label_ts = [el for i, el in enumerate(label_ts) if i in hand_LS_labels]
        back_LS_label_ts = [el for i, el in enumerate(label_ts) if i in back_LS_labels]

    ##############################################################################################
    # Return labels and pain ratings
    hand_all_label_ts = (
        (hand_NS_label_ts, hand_LS_label_ts, hand_HS_label_ts)
        if include_LS
        else (hand_NS_label_ts, hand_HS_label_ts)
    )
    back_all_label_ts = (
        (back_NS_label_ts, back_LS_label_ts, back_HS_label_ts)
        if include_LS
        else (back_NS_label_ts, back_HS_label_ts)
    )

    hand_all_ratings = (
        (hand_NS_ratings, hand_LS_ratings, hand_HS_ratings)
        if include_LS
        else (hand_NS_ratings, hand_HS_ratings)
    )
    back_all_ratings = (
        (back_NS_ratings, back_LS_ratings, back_HS_ratings)
        if include_LS
        else (back_NS_ratings, back_HS_ratings)
    )

    return hand_all_label_ts, back_all_label_ts, hand_all_ratings, back_all_ratings


def compute_connectivity_epochs(
    label_ts,
    roi_names,
    method,
    fmin,
    fmax,
    tmin,
    tmax,
    sfreq,
):
    con_epochs = mne_conn.spectral_connectivity_epochs(
        label_ts,
        method=method,
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_adaptive=True,
        n_jobs=1,
    )
    print(f"*con_epochs shape = {con_epochs.shape}*")
    return con_epochs


def compute_connectivity_resting_state(
    label_ts,
    roi_names,
    method,
    fmin,
    fmax,
    tmin,
    tmax,
    sfreq,
):
    # Change shape of resting state label_ts to 3-d for compatibility
    data = np.expand_dims(label_ts, axis=0)

    # Provide the freq points
    freqs = np.linspace(fmin, fmax, int((fmax - fmin) * 4 + 1))

    # This function does not support wpli2_debiased, so change to wpli instead
    if method == "wpli2_debiased":
        method = "wpli"

    con = mne_conn.spectral_connectivity_time(
        data=data,
        freqs=freqs,
        method=method,
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        n_jobs=1,
    )
    print(f"*con shape = {con.shape}*")
    return con


def compute_sub_avg_con(
    sub_id,
    group_name,
    processed_data_path,
    zscored_epochs_data_path,
    EO_resting_data_path,
    EC_resting_data_path,
    connectivity_methods,
    conditions,
    roi_names,
    Freq_Bands,
    tmin,
    tmax,
    sfreq,
    include_LS=False,
):
    """
    Compute the average connectivity for each subject, group, and condition.

    Args:
        sub_id (str): The subject ID.
        group_name (str): The name of the group.
        processed_data_path (str): The path to the processed data.
        zscored_epochs_data_path (str): The path to the z-scored epochs data.
        EO_resting_data_path (str): The path to the EO resting data.
        EC_resting_data_path (str): The path to the EC resting data.
        connectivity_methods (list): List of connectivity methods to compute.
        conditions (list): List of conditions.
        roi_names (list): List of regions of interest names.
        Freq_Bands (dict): Dictionary of frequency bands.
        tmin (float): The minimum time.
        tmax (float): The maximum time.
        sfreq (float): The sampling frequency.

    Returns:
        dict: A dictionary containing the connectivity results for each condition, method, and frequency band.
    """

    # Initialize dictionary for this subject
    results = {}

    # Separate epochs by stimulus
    (
        hand_all_label_ts,
        back_all_label_ts,
        hand_all_ratings,
        back_all_ratings,
    ) = separate_epochs_by_stim(
        sub_id, processed_data_path, zscored_epochs_data_path, include_LS=include_LS
    )

    # Resting state
    EO_filepath = os.path.join(EO_resting_data_path, f"{sub_id}_eyes_open.pkl")
    EC_filepath = os.path.join(EC_resting_data_path, f"{sub_id}_eyes_closed.pkl")
    label_ts_EO = utils.unpickle_data(EO_filepath)
    label_ts_EC = utils.unpickle_data(EC_filepath)

    # Unpack label_ts for each site and stimulus level
    label_ts_all = [*hand_all_label_ts, *back_all_label_ts]
    label_ts_all.extend([label_ts_EO, label_ts_EC])

    # Get the frequency bands
    fmins = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
    fmaxs = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])
    fmins = [Freq_Bands[f][0] for f in Freq_Bands]
    fmaxs = [Freq_Bands[f][1] for f in Freq_Bands]

    # Compute connectivity for epochs
    for method in connectivity_methods:
        for label_ts, condition in zip(label_ts_all, conditions):
            num_epochs = len(label_ts)
            if num_epochs == 0:
                continue
            for fmin, fmax, band_name in zip(fmins, fmaxs, Freq_Bands):
                table = [
                    ["Subject", sub_id],
                    ["Condition", condition],
                    ["Num. of epochs", len(label_ts)],
                    ["Band", band_name],
                    ["Method", method],
                ]
                print(tabulate(table, tablefmt="grid"))
                if isinstance(label_ts, list):
                    con = compute_connectivity_epochs(
                        label_ts,
                        roi_names,
                        method,
                        fmin,
                        fmax,
                        tmin,
                        tmax,
                        sfreq,
                    )
                    # reshape to roi x roi
                    con_data = con.get_data()
                    con_data = con_data.reshape(len(roi_names), len(roi_names))

                else:
                    # Compute connectivity for resting state
                    con = compute_connectivity_resting_state(
                        label_ts,
                        roi_names,
                        method,
                        fmin,
                        fmax,
                        tmin,
                        tmax,
                        sfreq,
                    )
                    # reshape to roi x roi
                    con_data = con.get_data()
                    con_data = con_data.reshape(len(roi_names), len(roi_names))

                print(f"*con_data shape = {con_data.shape}*")

                # Add result to dictionary
                if condition not in results:
                    results[condition] = {}
                if "num_epochs" not in results[condition]:
                    results[condition]["num_epochs"] = num_epochs
                if method not in results[condition]:
                    results[condition][method] = {}
                results[condition][method][band_name] = con_data
    return results


def compute_group_con(sub_con_dict, conditions, con_methods, band_names):
    """
    Compute the average connectivity for all subjects in each group, condition, method, and band.

    Args:
        sub_con_dict (dict): The dictionary containing the connectivity results for each subject.

    Returns:
        dict: A dictionary containing the average connectivity results for each group, condition, method, and band.
    """

    # Initialize dictionary for the averages
    avg_dict = {}

    # Get the list of subjects
    subjects = list(sub_con_dict.keys())

    # Iterate over all conditions, methods, and band names
    for condition in conditions:
        for method in con_methods:
            for band in band_names:
                # Compute the average for all subjects
                avg = np.mean(
                    [
                        sub_con_dict[subject][condition][method][band]
                        for subject in subjects
                    ],
                    axis=0,
                )

                # Add result to dictionary
                if condition not in avg_dict:
                    avg_dict[condition] = {}
                if method not in avg_dict[condition]:
                    avg_dict[condition][method] = {}
                avg_dict[condition][method][band] = avg

        # Sum the number of epochs in each condition
        num_epochs = np.sum(
            [sub_con_dict[subject][condition]["num_epochs"] for subject in subjects]
        )
        if "num_epochs" not in avg_dict[condition]:
            avg_dict[condition]["num_epochs"] = num_epochs

    return avg_dict


# plt.rcParams["font.size"] = 21


def plot_connectivity(
    con_data,
    method,
    band,
    roi_names,
    group_name,
    condition,
    num_epochs,
    save_path,
):
    # Plot parameters
    vmin, vmax = 0.0, 1.0
    cmap = "inferno"

    # Epochs uses wpli2_debiased while resting state uses wpli. Change to wpli in title as an umbrella term
    if method == "wpli2_debiased":
        method = "wpli"

    im = plt.imshow(
        con_data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.colorbar(im, label="Connectivity", cmap=cmap)

    plt.ylabel("Regions", labelpad=20)
    plt.yticks(range(len(roi_names)), labels=roi_names)

    plt.xlabel("Regions", labelpad=20)
    plt.xticks(range(len(roi_names)), labels=roi_names, rotation=45, ha="right")

    plt.title(
        f"Connectivity of {group_name} Group {condition} condition in {band} band ({method} method, {num_epochs} trials)"
    )
    filename = f"conn_{group_name}_{condition}_{band}_{method}.png"
    plt.savefig(os.path.join(save_path, filename), bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


def plot_connectivity_circle(
    con_data, method, band, roi_names, group_name, condition, num_epochs, save_path
):
    # Convert ROI names to labels
    labels = [
        mne.read_labels_from_annot(subject, regexp=roi, subjects_dir=subjects_dir)[0]
        for roi in roi_names
    ]
    # read colors
    node_colors = [label.color for label in labels]

    # We reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]
    lh_labels = [name for name in label_names if name.endswith("lh")]
    rh_labels = [name for name in label_names if name.endswith("rh")]

    # Get the y-location of the label
    label_ypos_lh = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos_lh.append(ypos)
    try:
        idx = label_names.index("Brain-Stem")
    except ValueError:
        pass
    else:
        ypos = np.mean(labels[idx].pos[:, 1])
        lh_labels.append("Brain-Stem")
        label_ypos_lh.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

    # For the right hemi
    rh_labels = [
        label[:-2] + "rh"
        for label in lh_labels
        if label != "Brain-Stem" and label[:-2] + "rh" in rh_labels
    ]

    # Save the plot order
    node_order = lh_labels[::-1] + rh_labels

    node_angles = mne.viz.circular_layout(
        label_names,
        node_order,
        start_pos=90,
        group_boundaries=[0, len(label_names) // 2],
    )

    # Epochs uses wpli2_debiased while resting state uses wpli. Change to wpli in title as an umbrella term
    if method == "wpli2_debiased":
        method = "wpli"

    # Plot parameters
    vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(
        figsize=(10, 8), facecolor="black", subplot_kw=dict(polar=True)
    )

    mne_conn.viz.plot_connectivity_circle(
        con_data,
        roi_names,
        title=f"Connectivity of {group_name} Group {condition} condition in {band} band ({method} method, {num_epochs} trials)",
        node_edgecolor="black",
        node_angles=node_angles,
        node_colors=node_colors,
        textcolor="white",
        fontsize_names=8,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    fig.tight_layout()
    filename = f"circle_{group_name}_{condition}_{band}_{method}.png"
    fig.savefig(
        os.path.join(save_path, filename),
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        dpi=300,
    )  # Save the figure
    plt.show()
    plt.close()


# def plot_global_connectivity(epochs, tmin, tmax, n_connections, con_epochs, Freq_Bands):
#     """
#     Plot the global connectivity over time.

#     Args:
#         epochs (Epochs): The epochs data.
#         tmin (float): The minimum time value to include in the plot.
#         tmax (float): The maximum time value to include in the plot.
#         n_connections (int): The number of connections.
#         con_epochs (list): The connectivity epochs.
#         Freq_Bands (dict): The frequency bands.
#     """

#     # Get the timepoints within the specified time range
#     times = epochs.times[(epochs.times >= tmin) & (epochs.times <= tmax)]

#     for c in range(len(con_epochs)):
#         # Get global average connectivity over all connections
#         con_epochs_raveled_array = con_epochs[c].get_data(output="raveled")
#         global_con_epochs = np.sum(con_epochs_raveled_array, axis=0) / n_connections

#         for i, (k, v) in enumerate(Freq_Bands.items()):
#             global_con_epochs_tmp = global_con_epochs[i]

#             # Get the timepoint with the highest global connectivity right after stimulus
#             t_con_max = np.argmax(global_con_epochs_tmp[times <= tmax])

#             # Plot the global connectivity
#             fig = plt.figure()
#             plt.plot(times, global_con_epochs_tmp)
#             plt.xlabel("Time (s)")
#             plt.ylabel(f"Global {k} wPLI over trials")
#             plt.title(f"Global {k} wPLI peaks {times[t_con_max]:.3f}s after stimulus")
