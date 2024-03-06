from eeg_toolkit import utils, preprocess
import mne_connectivity as mne_conn
from mne_connectivity import envelope_correlation
import mne
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import numpy as np
from tabulate import tabulate
import scipy.stats as stats
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
    sub_id, processed_data_path, stc_data_path, include_LS=True, pain_thresh=None
):
    """
    Separates epochs by stimulus for a given subject.

    :param sub_id: The ID of the subject.
    :param processed_data_path: The path to the processed data.
    :param stc_data_path: The path to the label time courses.
    :param pain_thresh: The pain vthresh. Default is None.

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

    label_ts = utils.unpickle_data(stc_data_path, f"{sub_id}_epochs.pkl")
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
        tmin=tmin,
        tmax=tmax,
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
    sfreq,
):
    # Change shape of resting state label_ts to 3-d for compatibility
    data = np.expand_dims(label_ts, axis=0)

    # Provide the freq points
    freqs = np.linspace(fmin, fmax, int((fmax - fmin) * 4 + 1))

    # This function does not support dwpli2_debiased, so change to dwpli instead
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


def bp_gen(
    label_ts,
    sfreq,
    fmin,
    fmax,
    tmin,
    tmax,
):
    """
    Generate band-pass filtered data using MNE library.

    :param label_ts: list of time series data
    :param sfreq: sampling frequency
    :param fmin: minimum frequency of the passband
    :param fmax: maximum frequency of the passband
    :return: generator yielding band-pass filtered data
    """
    for ts in label_ts:
        # crop the data between tmin and tmax
        ts = ts[..., int(np.round(tmin * sfreq)) : int(np.round(tmax * sfreq))]

        print(f"ts shape = {ts.shape}")
        yield mne.filter.filter_data(
            ts, sfreq, fmin, fmax, phase="zero-double", method="iir"
        )


def compute_aec(method, label_ts, sfreq, fmin, fmax, tmin, tmax, roi_names):
    """
    Compute the correlation between regions of interest (ROIs) using different methods.

    Parameters:
    - method (str): The method used for computing the correlation. It can be "aec_pairwise" or "aec_symmetric".
    - label_ts (array-like): The timeseries data for the ROIs.
    - sfreq (float): The sampling frequency of the timeseries data.
    - fmin (float): The minimum frequency of interest for the correlation computation.
    - fmax (float): The maximum frequency of interest for the correlation computation.
    - roi_names (list): The names of the ROIs.

    Returns:
    - data (array): The computed correlation data reshaped to match the ROI names.
    """
    corr = None

    # for resting data
    print(f"label_ts shape =  {np.asarray(label_ts).shape}")
    if np.asarray(label_ts).ndim == 1:
        label_ts = [np.expand_dims(np.asarray(label_ts), axis=0)]

    if method == "aec_pairwise":
        corr_obj = envelope_correlation(
            bp_gen(label_ts, sfreq, fmin, fmax, tmin, tmax),
            orthogonalize="pairwise",
        )
        corr = corr_obj.combine()
        corr = corr.get_data(output="dense")[:, :, 0]
    if method == "aec_symmetric":
        label_ts_orth = mne_conn.envelope.symmetric_orth(label_ts)
        corr_obj = envelope_correlation(
            bp_gen(label_ts_orth, sfreq, fmin, fmax, tmin, tmax), orthogonalize=False
        )
        corr = corr_obj.combine()
        corr = corr.get_data(output="dense")[:, :, 0]
        corr.flat[:: corr.shape[0] + 1] = 0  # vzero out the diagonal
        corr = np.abs(corr)

    return corr


def plot_corr(corr, title):
    """
    Plots a correlation matrix.

    Args:
        corr (array-like): The correlation matrix to be plotted.
        title (str): The title of the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(corr, cmap="viridis", clim=np.percentile(corr, [5, 95]))
    fig.suptitle(title)


def plot_degree(corr, title, labels, inv):
    """
    Plot the degree of connectivity in the brain network.

    Parameters:
    corr (array-like): The connectivity matrix.
    title (str): The title of the plot.
    labels (Label): The labels for regions of interest.
    inv (dict): The inverse operator.

    Returns:
    instance of mne.viz.Brain: The plot of the degree connectivity.
    """
    vthresh_prop = 0.15  # percentage of strongest edges to keep in the graph
    degree = mne_conn.degree(corr, vthresh_prop=vthresh_prop)
    stc = mne.labels_to_stc(labels, degree)
    stc = stc.in_label(
        mne.Label(inv["src"][0]["vertno"], hemi="lh")
        + mne.Label(inv["src"][1]["vertno"], hemi="rh")
    )
    return stc.plot(
        clim=dict(kind="percent", lims=[75, 85, 95]),
        colormap="gnuplot",
        subjects_dir=subjects_dir,
        views="dorsal",
        hemi="both",
        smoothing_steps=25,
        time_label=title,
    )


def compute_sub_avg_con(
    sub_id,
    group_name,
    processed_data_path,
    zscored_epochs_data_path,
    EO_resting_data_path,
    EC_resting_data_path,
    con_methods,
    conditions,
    condition_dict,
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
        con_methods (list): List of connectivity methods to compute.
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
    ) = separate_epochs_by_stim(sub_id, processed_data_path, zscored_epochs_data_path)

    # Resting state
    label_ts_EO = utils.unpickle_data(EO_resting_data_path, f"{sub_id}_eyes_open.pkl")
    label_ts_EC = utils.unpickle_data(EC_resting_data_path, f"{sub_id}_eyes_closed.pkl")

    # Unpack label_ts for each site and stimulus level
    label_ts_all = [*hand_all_label_ts, *back_all_label_ts]
    label_ts_all.extend([label_ts_EO, label_ts_EC])

    # Get the frequency bands
    fmins = [Freq_Bands[f][0] for f in Freq_Bands]
    fmaxs = [Freq_Bands[f][1] for f in Freq_Bands]

    # Use only label_ts from overlap of condition_dict and conditions
    desired_conditions_ids = [v for k, v in condition_dict.items() if k in conditions]
    desired_label_ts = [label_ts_all[i] for i in desired_conditions_ids]

    # Compute connectivity for epochs
    for method in con_methods:
        for label_ts, condition in zip(desired_label_ts, conditions):
            # Adjust for resting state
            if "Eyes" in condition:
                label_ts_new = [np.array(lst) for lst in label_ts]
                label_ts = label_ts_new

            # If label_ts contains NaN values, print and break loop.
            if np.isnan(label_ts).any():
                print(
                    f"Skipping {method} {condition} for {sub_id} due to NaN values in label_ts."
                )
                continue

            num_epochs = len(label_ts)
            if num_epochs == 0:
                continue
            for fmin, fmax, band_name in zip(fmins, fmaxs, Freq_Bands):
                table = [
                    ["Subject", sub_id],
                    ["Condition", condition],
                    [
                        "Num. of epochs",
                        np.array(label_ts).shape[0] if "Eyes" not in condition else 1,
                    ],
                    ["Band", band_name],
                    ["Method", method],
                ]
                print(tabulate(table, tablefmt="grid"))

                ## Amplitude Envelope Correlation
                if method == "aec_pairwise":
                    if "Eyes" in condition and np.array(label_ts).ndim < 3:
                        label_ts_arr = np.array(label_ts)
                        label_ts = np.expand_dims(label_ts_arr, axis=0)
                    # Compute correlation
                    corr = compute_aec(
                        "aec_pairwise",
                        label_ts,
                        sfreq,
                        fmin,
                        fmax,
                        tmin=tmin,
                        tmax=tmax,
                        roi_names=roi_names,
                    )
                    data = corr.reshape(len(roi_names), len(roi_names))
                elif method == "aec_symmetric":
                    if "Eyes" in condition and np.array(label_ts).ndim < 3:
                        label_ts_arr = np.array(label_ts)
                        label_ts = np.expand_dims(label_ts_arr, axis=0)
                    # Compute correlation
                    corr = compute_aec(
                        "aec_symmetric",
                        label_ts,
                        sfreq,
                        fmin,
                        fmax,
                        tmin=tmin,
                        tmax=tmax,
                        roi_names=roi_names,
                    )
                    data = corr.reshape(len(roi_names), len(roi_names))

                elif "Eyes" not in condition and "aec" not in method:
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
                    data = con.get_data()
                    data = data.reshape(len(roi_names), len(roi_names))

                elif "Eyes" in condition and "aec" not in method:
                    # Compute connectivity for resting state
                    con = compute_connectivity_resting_state(
                        label_ts,
                        roi_names,
                        method,
                        fmin,
                        fmax,
                        sfreq,
                    )
                    # reshape to roi x roi
                    data = con.get_data()
                    data = data.reshape(len(roi_names), len(roi_names))

                print(f"*data shape = {data.shape}*")

                # Add result to dictionary
                if condition not in results:
                    results[condition] = {}
                if "num_epochs" not in results[condition]:
                    results[condition]["num_epochs"] = num_epochs
                if method not in results[condition]:
                    results[condition][method] = {}
                results[condition][method][band_name] = data
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
                stack = np.stack(
                    [
                        sub_con_dict[subject][condition][method][band]
                        for subject in subjects
                    ],
                )
                # Add result to dictionary
                if condition not in avg_dict:
                    avg_dict[condition] = {}
                if method not in avg_dict[condition]:
                    avg_dict[condition][method] = {}
                avg_dict[condition][method][band] = stack

        # Sum the number of epochs in each condition
        num_epochs = np.sum(
            [sub_con_dict[subject][condition]["num_epochs"] for subject in subjects]
        )
        if "num_epochs" not in avg_dict[condition]:
            avg_dict[condition]["num_epochs"] = num_epochs

    return avg_dict


plt.rcParams["font.size"] = 13


def get_method_plot_name(method):
    """
    Returns the plot name based on the method provided.

    Args:
        method (str): The method for which the plot name is to be retrieved.

    Returns:
        str: The plot name corresponding to the provided method, or the uppercase method if no match is found.
    """
    method_dict = {
        "wpli2_debiased": "dwPLI",
        "dpli": "dPLI",
        "aec_pairwise": "AEC Pairwise",
        "aec_symmetric": "AEC Symmetric",
    }

    return method_dict.get(method, method.upper())



def mann_whitney_test(
    group1_stack, group2_stack, roi_names, round_neg_vals=True, method=None
):
    """
    Perform Mann-Whitney U test on group1_stack and group2_stack for each ROI combination.
    Calculate p-values, means, and standard error of the mean.
    Args:
        group1_stack: 3D array of data for group 1
        group2_stack: 3D array of data for group 2
        roi_names: List of names for the regions of interest
        method: Method for adjusting the data (default is None)
    Returns:
        p_values: Array of p-values for each ROI combination
        means_1: Means of group 1 data for each ROI combination
        sem_1: Standard error of the mean of group 1 data for each ROI combination
        means_2: Means of group 2 data for each ROI combination
        sem_2: Standard error of the mean of group 2 data for each ROI combination
    """
    n = len(roi_names)
    p_values = np.zeros((n, n))
    means_1 = np.zeros((n, n))
    means_2 = np.zeros((n, n))
    sem_1 = np.zeros((n, n))
    sem_2 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Perform Mann-Whitney U test
            data1 = group1_stack[:, i, j]
            data2 = group2_stack[:, i, j]

            # Round negative values
            if round_neg_vals:
                data1[data1 < 0] = 0
                data2[data2 < 0] = 0

            # If method is 'dpli', adjust data
            if method == "dpli":
                data1_tmp = np.abs(data1 - 0.5)
                data2_tmp = np.abs(data2 - 0.5)

                u, p = stats.mannwhitneyu(data1_tmp, data2_tmp)
                p_values[i, j] = p
            else:
                u, p = stats.mannwhitneyu(data1, data2)
                p_values[i, j] = p

            # Calculate means
            means_1[i, j] = np.mean(data1)
            means_2[i, j] = np.mean(data2)

            # Calculate SEM
            sem_1[i, j] = stats.sem(data1)
            sem_2[i, j] = stats.sem(data2)

    return p_values, means_1, sem_1, means_2, sem_2


def plot_connectivity_circle(
    data,
    method,
    band,
    roi_names,
    roi_acronyms,
    condition,
    save_path,
    vmin=None,
    vmax=None,
    fig=None,
    subplot=None,
    colormap="YlGnBu",
    title_prefix=None,
    save_fig=False,
):
    """
    Plot the connectivity circle for the given connectivity data.

    Args:
        data (numpy.ndarray): The connectivity data.
        method (str): The method used for connectivity estimation.
        band (str): The frequency band used for connectivity estimation.
        roi_names (list): The names of the regions of interest.
        group_name (str): The name of the group.
        condition (str): The condition of the data.
        num_epochs (int): The number of epochs.
        save_path (str): The path to save the plot.

    Returns:
        None
    """
    # Convert ROI names to labels
    labels = [
        mne.read_labels_from_annot(
            subject, regexp=roi, subjects_dir=subjects_dir, verbose=False
        )[0]
        for roi in roi_names
    ]
    # read colors
    node_colors = [label.color for label in labels]

    # We reorder the labels based on their location in the left hemi
    # label_names = [label.name for label in labels]
    label_names = roi_acronyms
    lh_labels = [name for name in label_names if name.endswith("lh")]
    rh_labels = [name for name in label_names if name.endswith("rh")]

    # Get the y-location of the label
    label_ypos_lh = list()
    for name in lh_labels:
        data_idx = label_names.index(name)
        ypos = np.mean(labels[data_idx].pos[:, 1])
        label_ypos_lh.append(ypos)
    try:
        data_idx = label_names.index("Brain-Stem")
    except ValueError:
        pass
    else:
        ypos = np.mean(labels[data_idx].pos[:, 1])
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

    # Circular layout
    node_angles = mne.viz.circular_layout(
        label_names,
        node_order,
        start_pos=90,
        group_boundaries=[0, len(label_names) // 2],
    )

    # Plot parameters
    vmin, vmax = (0.0, 0.7) if method == "dwpli" else (None, None)
    vmin, vmax = (0.0, 0.5) if method == "dpli" else (None, None)
    vmin, vmax = (-1.0, 1.0) if "aec" in method else (None, None)

    mne_conn.viz.plot_connectivity_circle(
        data,
        roi_acronyms,
        node_edgecolor="white",
        node_angles=node_angles,
        node_colors=node_colors,
        textcolor="black",
        facecolor="white",
        colormap=colormap,
        fontsize_names=8,
        vmin=vmin,
        vmax=vmax,
        fig=fig,
        subplot=subplot,
        show=False,
    )

    # Save figure
    if save_fig:
        fig.tight_layout()
        filename = f"circle_{condition}_{band}_{method}.png"
        fig.savefig(
            os.path.join(save_path, filename),
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            dpi=300,
        )
    # plt.show()
    # plt.close()

def plot_connectivity_and_stats(
    means_1,
    means_2,
    sem_1,
    sem_2,
    p_values,
    nepochs,
    group_names,
    method,
    band,
    roi_names,
    condition,
    titles,
    save_names,
    save_path,
    fig=None,
    subplot=None,
    roi_acronyms=None,
    save_fig=True,
    highlight_pvals=True,
    show_only_significant=True,
    min_fc_val=None,  # optional minimum value to highlight
    set_title=True,
    show_fc_vals=True,
    round_neg_vals=False,
):

    # Set min_fc_val if not provided
    if min_fc_val is None:
        min_fc_val = -1

    # Get highlight indices
    highlight_ij = []
    for i in range(len(roi_names)):
        for j in range(len(roi_names)):
            if p_values[i, j] < 0.05:
                highlight_ij.append((i, j))

    # Make top-right diagonal and above white
    for i in range(len(roi_names)):
        for j in range(i, len(roi_names)):
            # Remove those from highlight_ij
            if (i, j) in highlight_ij:
                highlight_ij.remove((i, j))

    # Create figure and indicate position of p-value plot
    # fig, axes = plt.subplots(1, 3, figsize=(12, 12))
    pval_pos = 2

    # Get shortened method name for plot
    method = get_method_plot_name(method)

    # Print table summary of mean and sem
    header = ["ROI Pair", "P-Value", "Mean ± SEM (1)", "Mean ± SEM (2)"]
    table = []
    for region_pair in highlight_ij:
        roi_pair = f"{roi_acronyms[region_pair[0]]} <-> {roi_acronyms[region_pair[1]]}"
        p_val = np.round(p_values[region_pair[0], region_pair[1]], 3)
        mean_sem_1 = f"{np.round(means_1[region_pair[0], region_pair[1]],3)} ± {np.round(sem_1[region_pair[0], region_pair[1]],3)}"
        mean_sem_2 = f"{np.round(means_2[region_pair[0], region_pair[1]],3)} ± {np.round(sem_2[region_pair[0], region_pair[1]],3)}"

        table.append([roi_pair, p_val, mean_sem_1, mean_sem_2])
    print(tabulate(table, headers=header, tablefmt="pretty"))

    # Choose the colormap
    colormap = "YlGnBu"

    # Loop through means and p values for plotting
    for (
        data_idx, 
        data, 
        # ax,
        ) in zip(
        range(3),
        [
            means_1,
            means_2,
            p_values,
        ],
        # axes,
    ):
            
        # TODO: temporary for fixing the tiny plot
        fig, ax = plt.subplots()
        axes = []*3
        axes[2] = ax
    
        # Plot parameters
        vmin, vmax = None, None
        if method == "dwPLI":
            vzero = 0.0
            vthresh = 0.5
            vmin, vmax = (
                (vzero, vzero + vthresh) if data_idx != pval_pos else (0.0, 1.0)
            )
        elif method == "dPLI":
            vzero = 0.5
            vthresh = 0.2
            vmin, vmax = (
                (vzero - vthresh, vzero + vthresh)
                if data_idx != pval_pos
                else (0.0, 1.0)
            )
        elif "AEC" in method:
            vzero = 0.0
            vthresh = 0.3
            vmin, vmax = (
                (vzero, vzero + vthresh) if data_idx != pval_pos else (0.0, 1.0)
            )

        # Plot circle for FC values, and connectivity matrix just for p-values
        im = None
        if data_idx == pval_pos:
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=colormap)
        else:
            plot_connectivity_circle(
                data=data,
                method=method,
                band=band,
                roi_names=roi_names,
                roi_acronyms=roi_acronyms,
                condition=condition,
                save_path=save_path,
                colormap=colormap,
                fig=fig,
                vmin=vmin,
                vmax=vmax,
                subplot=(1, 3, data_idx + 1),
                title_prefix=None,
                save_fig=False,
            )

        # Overlay values
        if data_idx != pval_pos:
            for i in range(len(roi_names)):
                for j in range(len(roi_names)):

                    # If round_neg_vals, round negative values
                    if round_neg_vals and data[i, j] < 0:
                        data[i, j] == 0.0

                    # Ignore really low values
                    if show_fc_vals:
                        if not np.isnan(data[i, j]) and data[i, j] > min_fc_val:
                            ax.text(
                                j,
                                i,
                                round(data[i, j], 3),
                                ha="center",
                                va="center",
                                color="k" if method == "dPLI" else "w",
                                fontsize=11,
                            )
        if data_idx == pval_pos:
            for i in range(len(roi_names)):
                for j in range(len(roi_names)):
                    if data[i, j] < 0.05 and not np.isnan(data[i, j]):
                        if show_fc_vals:
                            ax.text(
                                j,
                                i,
                                round(data[i, j], 3),
                                ha="center",
                                va="center",
                                color="w",
                                fontsize=11,
                            )

        # Add rectangles for highlighted squares
        if highlight_pvals:
            for i, j in highlight_ij:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )

        if im and data_idx != 1:  # skip the first plot
            label_text = "Connectivity" if data_idx != pval_pos else "p-value"
            cmap = plt.get_cmap(colormap)
            plt.colorbar(im, label=label_text, cmap=cmap)

        axes[2].set_ylabel("Regions", labelpad=20)
        axes[2].set_yticks(range(len(roi_names)), labels=roi_names)

        axes[2].set_xlabel("Regions", labelpad=20)
        axes[2].set_xticks(
            range(len(roi_names)), labels=roi_names, rotation=45, ha="right"
        )

        if set_title:
            if data_idx != pval_pos:  # group 1 or group 2
                ax.set_title(
                    f"{titles[data_idx]} | {condition} | {band} | ({method} method, {nepochs[data_idx-1]} trials)"
                )
            else:  # p-values
                ax.set_title(
                    f"{titles[data_idx]} | {condition} | {band} | ({method} method, {nepochs[0]} vs. {nepochs[1]} trials)"
                )

    # Make top-right diagonal and above white
    for i in range(len(roi_names)):
        for j in range(i, len(roi_names)):
            data[i, j] = np.nan

    # If showing only significant values, make them appear white
    if show_only_significant:
        for i in range(len(roi_names)):
            for j in range(len(roi_names)):
                if p_values[i, j] >= 0.05:
                    data[i, j] = np.nan
                    p_values[i, j] = np.nan

    filename = f"{condition}_{band}_{method}.png"
    if save_fig:
        fig.savefig(os.path.join(save_path, filename), bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()
