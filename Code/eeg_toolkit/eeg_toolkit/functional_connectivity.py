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
from collections import defaultdict
import bct

fs_dir = fetch_fsaverage(verbose=True)
subject = "fsaverage"
subjects_dir = os.path.dirname(fs_dir)

# Font size setting
plt.rcParams["font.size"] = 13


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
    :param pain_thresh: The pain vtolerance. Default is None.

    Returns:
    - hand_all_label_tsF: A tuple containing the label time courses for hand stimuli.
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

    # Provide the connections points
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


def compute_aec(method, 
                label_ts, 
                sfreq, 
                fmin, 
                fmax, 
                tmin, 
                tmax, 
                roi_names,
                orthogonalize_AEC=True):
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
        if orthogonalize_AEC:
            label_ts_orth = mne_conn.envelope.symmetric_orth(label_ts)
        else:
            label_ts_orth = label_ts
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
    threshold_prop = 0.15  # percentage of strongest edges to keep in the graph
    degree = mne_conn.degree(corr, threshold_prop=threshold_prop)
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
    roi_acronyms,
    Freq_Bands,
    sfreq,
    orthogonalize_AEC=True,
    left_pain_ids=None,
    right_pain_ids=None,
    bilateral_pain_ids=None,
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
        dict: A dictionary containing the connectivity sub_con_dict for each condition, method, and frequency band.
    """
    # Set tmax 
    tmin = 0.0
    tmax_epo = 1.25  # exclude the baseline period for connectivity estimation 
    tmax_resting = 5*60 # 5 minutes resting eyes open
    
    # Initialize dictionary for this subject
    sub_con_dict = {}

    if zscored_epochs_data_path is not None:
        # Separate epochs by stimulus
        (
            hand_all_label_ts,
            back_all_label_ts,
            hand_all_ratings,
            back_all_ratings,
        ) = separate_epochs_by_stim(sub_id, processed_data_path, zscored_epochs_data_path)
    else:
        hand_all_label_ts = []
        back_all_label_ts = []
        evoked_data_flag = False
        
    # Resting state
    label_ts_EO = utils.unpickle_data(EO_resting_data_path, f"{sub_id}_eyes_open.pkl")

    # tmax_resting = len(label_ts_EO[0]) 

    # label_ts_EC = utils.unpickle_data(EC_resting_data_path, f"{sub_id}_eyes_closed.pkl")

    ##############################################################################################
    # Check laterality and adjust order of label time courses
    s1_lh_index = roi_acronyms.index("S1-lh")
    s1_rh_index = roi_acronyms.index("S1-rh")
    if left_pain_ids is not None and sub_id in left_pain_ids:
        print("Left pain, -rh is already contralateral")
    elif right_pain_ids is not None and sub_id in right_pain_ids:
        print("Right pain, -rh is now contralateral")
        # Swap label_ts[2] and label_ts[8]
        label_ts_EO[[s1_lh_index, s1_rh_index]] = label_ts_EO[
            [s1_rh_index, s1_lh_index]
        ]
        # label_ts_EC[[s1_lh_index, s1_rh_index]] = label_ts_EC[
        #     [s1_rh_index, s1_lh_index]
        # ]
    elif bilateral_pain_ids is not None and sub_id in bilateral_pain_ids:
        print("Bilateral pain, -lh and -rh have been combined into contralateral")
        # Average -lh and -rh for S1 and set -rh to contralateral.
        avg_S1_EO = (label_ts_EO[s1_lh_index] + label_ts_EO[s1_rh_index]) / 2
        # avg_S1_EC = (label_ts_EC[s1_lh_index] + label_ts_EC[s1_rh_index]) / 2
        # Set contralateral as average. Do not alter S1-lh to avoid rank deficiency
        label_ts_EO[s1_rh_index] = avg_S1_EO
        # label_ts_EC[s1_rh_index] = avg_S1_EC
        # In next steps, bilateral subjects will be excluded from contributing S1-i data to the group stack
        
    # Unpack label_ts for each site and stimulus level
    if evoked_data_flag:
        label_ts_all = [*hand_all_label_ts, *back_all_label_ts]
        label_ts_all.extend([label_ts_EO, 
                            #  label_ts_EC,
                            ])
    else:
        label_ts_all = [label_ts_EO]

    # Get the frequency bands
    fmins = [Freq_Bands[f][0] for f in Freq_Bands]
    fmaxs = [Freq_Bands[f][1] for f in Freq_Bands]

    # Use only label_ts from overlap of condition_dict and conditions
    
    if not evoked_data_flag: # change conditions_dict to include only resting condition
        condition_dict = {
            "Eyes Open": 0,
        }
        
    desired_conditions_ids = [v for k, v in condition_dict.items() if k in conditions]
    desired_label_ts = [label_ts_all[i] for i in desired_conditions_ids]

    # Compute connectivity for epochs
    for label_ts, condition in zip(desired_label_ts, conditions):
        # Set up the first level of the dictionary
        sub_con_dict[condition] = {}
        
        # Set tmax based on condition
        tmax = tmax_epo if condition == "Hand 256 mN" else tmax_resting
        
        for method in con_methods:
            # Set up the second level of the dictionary
            num_epochs = len(label_ts)
            if num_epochs == 0:
                continue
            sub_con_dict[condition][method] = {}
            sub_con_dict[condition]["num_epochs"] = num_epochs

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

            for fmin, fmax, band_name in zip(fmins, fmaxs, Freq_Bands):
                # Set up the third level of the dictionary
                sub_con_dict[condition][method][band_name] = {}

                # Ignore some specific condition/method combinations
                if condition == "Hand 256 mN" and "aec" in method:
                    print(f"Skipping {method} {condition} for {sub_id}.")
                    sub_con_dict[condition][method][band_name]["data"] = dict()
                    sub_con_dict[condition][method][band_name]["top 3"] = dict()
                    continue
                elif condition == "Eyes Open" and method == "wpli2_debiased":
                    print(f"Skipping {method} {condition} for {sub_id}.")
                    sub_con_dict[condition][method][band_name]["data"] = dict()
                    sub_con_dict[condition][method][band_name]["top 3"] = dict()
                    continue
                else:
                    pass

                print(f"\nComputing {method} {condition} {band_name} for {sub_id}...")

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
                        orthogonalize_AEC=orthogonalize_AEC,
                    )
                    data = corr.reshape(label_ts.shape[1], label_ts.shape[1])
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
                        orthogonalize_AEC=orthogonalize_AEC,
                    )
                    print(corr.shape)
                    data = corr.reshape(label_ts.shape[1], label_ts.shape[1])
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
                    data = corr.reshape(label_ts.shape[1], label_ts.shape[1])
                print(f"*data shape = {data.shape}*")

                # Add result to dictionary
                sub_con_dict[condition][method][band_name]["data"] = data

                # Top 3 connections and their strengths
                top_connections, strength = get_top_connections(data, method, n_top=3)
                sub_con_dict[condition][method][band_name]["top 3"] = top_connections

    return sub_con_dict


def compute_group_con(
    sub_con_dict,
    conditions,
    con_methods,
    band_names,
    left_pain_ids=None,
    right_pain_ids=None,
    bilateral_pain_ids=None,
):
    """
    Compute the average connectivity for all subjects in each group, condition, method, and band.

    Args:
        sub_con_dict (dict): The dictionary containing the connectivity sub_con_dict for each subject.

    Returns:
        dict: A dictionary containing the average connectivity sub_con_dict for each group, condition, method, and band.
    """

    # Initialize dictionary for the averages
    group_dict = {}

    # Get the list of subjects
    # Contralateral case
    if (
        left_pain_ids is not None
        and right_pain_ids is not None
        and bilateral_pain_ids is not None
    ):
        subjects = (
            set(sub_con_dict.keys())
            & set(left_pain_ids + right_pain_ids)
            & set(bilateral_pain_ids)
        )
    else:
        subjects = list(sub_con_dict.keys())

    # Iterate over all conditions, methods, and band names
    for condition in conditions:
        # Initialize dictionary for the condition
        group_dict[condition] = {}
        for method in con_methods:
            # Skip hand+aec and resting+pli
            if condition == "Hand 256 mN" and "aec" in method:
                continue
            elif condition == "Eyes Open" and method == "wpli2_debiased":
                continue

            # Initialize dictionary for the method
            group_dict[condition][method] = {}

            for band in band_names:
                # Initialize dictionary for the band
                group_dict[condition][method][band] = {}

                # Initialize a list to hold data arrays
                data_to_stack = []

                for i, subject in enumerate(subjects):
                    # Check if the data exists before adding it to the list
                    if isinstance(
                        sub_con_dict[subject][condition][method][band]["data"],
                        np.ndarray,
                    ):
                        data_to_stack.append(
                            sub_con_dict[subject][condition][method][band]["data"]
                        )

                # Add result to dictionary
                group_dict[condition][method][band]["data"] = np.stack(
                    np.array(data_to_stack)
                )

                # Find the top 3 connections that occur most frequently
                top_3_connections = [
                    sub_con_dict[subject][condition][method][band]["top 3"]
                    for subject in subjects
                    if isinstance(
                        sub_con_dict[subject][condition][method][band]["top 3"], list
                    )
                ]
                # Initialize a defaultdict to store the connections and their strengths
                connections_dict = defaultdict(list)

                # Loop through the data
                for sublist in top_3_connections:
                    for connection, strength in sublist:
                        # Append the strength to the corresponding connection in the dictionary
                        connections_dict[connection].append(strength)

                # Sort the connections by their frequency and select the top 3
                top_3_connections = sorted(
                    connections_dict,
                    key=lambda k: len(connections_dict[k]),
                    reverse=True,
                )[:3]

                # Store the top 3 connections and their strengths in the group_dict
                group_dict[condition][method][band]["top 3"] = {}
                group_dict[condition][method][band]["top 3"]["connections"] = [
                    connection for connection in top_3_connections
                ]
                group_dict[condition][method][band]["top 3"]["frequency"] = [
                    len(connections_dict[connection])
                    for connection in top_3_connections
                ]
                group_dict[condition][method][band]["top 3"]["mean strength"] = [
                    np.mean(connections_dict[connection]).round(3)
                    for connection in top_3_connections
                ]

        # Sum the number of epochs in each condition
        num_epochs = np.sum(
            [sub_con_dict[subject][condition]["num_epochs"] for subject in subjects]
        )
        if "num_epochs" not in group_dict[condition]:
            group_dict[condition]["num_epochs"] = num_epochs

    print("Group connectivity completed.")

    return group_dict


def get_method_plot_name(method):
    method_dict = {
        "wpli2_debiased": "dwPLI",
        "dpli": "dPLI",
        "aec_pairwise": "AEC Pairwise",
        "aec_symmetric": "AEC Symmetric",
    }

    return method_dict.get(method, method.upper())


def mann_whitney_test(
    group1_stack, 
    group2_stack, 
    roi_acronyms,
    sub_ids1=None,
    sub_ids2=None,
    condition=None,
    bilateral_pain_ids=None,
    round_neg_vals=True, 
    method=None
):
    """
    Perform Mann-Whitney U test on group1_stack and group2_stack for each ROI combination.
    Calculate p-values, means, and standard error of the mean.
    Args:
        group1_stack: 3D array of data for group 1
        group2_stack: 3D array of data for group 2
        roi_acronyms: List of names for the regions of interest
        method: Method for adjusting the data (default is None)
    Returns:
        p_values: Array of p-values for each ROI combination
        means_1: Means of group 1 data for each ROI combination
        sem_1: Standard error of the mean of group 1 data for each ROI combination
        means_2: Means of group 2 data for each ROI combination
        sem_2: Standard error of the mean of group 2 data for each ROI combination
    """

    # Initialize arrays for p-values, means, and standard error of the mean
    n = len(roi_acronyms)
    p_values = np.zeros((n, n))
    means_1 = np.zeros((n, n))
    means_2 = np.zeros((n, n))
    sem_1 = np.zeros((n, n))
    sem_2 = np.zeros((n, n))

    # Get indices for S1-i and S1-c
    if 'Eyes' in condition and bilateral_pain_ids is not None:
        s1_lh_index = roi_acronyms.index("S1-i")
    
        # Check if the subject is in the bilateral pain group
        ignore_inds = np.where(np.in1d(sub_ids1, bilateral_pain_ids))[0]
        
        group1_stack[ignore_inds, s1_lh_index, :] = np.nan
        group1_stack[ignore_inds, :, s1_lh_index] = np.nan

        # Repeat for group 2
        ignore_inds = np.where(np.in1d(sub_ids2, bilateral_pain_ids))[0]
        group2_stack[ignore_inds, s1_lh_index, :] = np.nan
        group2_stack[ignore_inds, :, s1_lh_index] = np.nan

    for i in range(n):
        for j in range(n):
            # Perform Mann-Whitney U test
            data1 = group1_stack[:, i, j]
            data2 = group2_stack[:, i, j]

            # Ignore NaN values from subjects without S1-i
            data1 = data1[~np.isnan(data1)]  
            data2 = data2[~np.isnan(data2)]
            
            # Round negative values
            if round_neg_vals:
                print(data1)
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


def make_symmetric(matrix):
    """
    A function that takes a matrix and makes it symmetric by copying the lower triangle to the upper triangle.
    Parameters:
    - matrix: a numpy array representing the input matrix
    Returns:
    - symmetric_matrix: a numpy array that is the symmetric version of the input matrix
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Copy the lower triangle to the upper triangle
    lower_triangle = np.tril(matrix, -1)
    upper_triangle = np.transpose(lower_triangle)
    symmetric_matrix = lower_triangle + upper_triangle

    return symmetric_matrix


def compute_centrality_and_test(
    group1_stack,
    group2_stack,
    roi_acronyms,
    condition,
    sub_ids1=None,
    sub_ids2=None,
    bilateral_pain_ids=None,
    method=None,
):
    """
    Compute centrality measures for group1_stack and group2_stack.
    """

    # Replace zero values with small value
    min_nonzero = np.min(group1_stack[group1_stack > 0])
    group1_stack[group1_stack <= 0] = min_nonzero

    min_nonzero = np.min(group2_stack[group2_stack > 0])
    group2_stack[group2_stack <= 0] = min_nonzero

    # Convert connectivity to connection-length matrix
    group1_stack = 1 / group1_stack
    group2_stack = 1 / group2_stack

    # Compute betweenness centrality for each region for each subject
    group1_centrality = []
    group2_centrality = []

    # Get indices for S1-i and S1-c
    if 'Eyes' in condition:
        s1_lh_index = roi_acronyms.index("S1-i")

    # For each subject, compute betweenness centrality
    for i in range(len(group1_stack)):
        # Make adjacency matrix symmetric first
        symm_1 = make_symmetric(group1_stack[i])
        symm_2 = make_symmetric(group2_stack[i])

        # Compute betweenness centrality
        if 'Eyes' not in condition or sub_ids1[i] not in bilateral_pain_ids:
            group1_centrality.append(bct.betweenness_wei(symm_1))
        else:
            sub_bc = bct.betweenness_wei(symm_1)
            sub_bc[s1_lh_index] = np.nan
            group1_centrality.append(sub_bc)

        # Repeat for group 2
        if 'Eyes' not in condition or sub_ids2[i] not in bilateral_pain_ids:
            group2_centrality.append(bct.betweenness_wei(symm_2))
        else:
            sub_bc = bct.betweenness_wei(symm_2)
            sub_bc[s1_lh_index] = np.nan
            group2_centrality.append(sub_bc)

        # Normalize betweenness centrality
        N = len(roi_acronyms)
        bc1 = group1_centrality[i]
        bc2 = group2_centrality[i]
        bc_norm1 = bc1 / ((N - 1) * (N - 2))
        bc_norm2 = bc2 / ((N - 1) * (N - 2))
        group1_centrality[i] = bc_norm1
        group2_centrality[i] = bc_norm2
        

    # Convert centrality lists to arrays
    group1_centrality = np.array(group1_centrality)
    group2_centrality = np.array(group2_centrality)
    
    # Perform Mann-Whitney U test between the nodes of both groups
    p_values = []
    means_1 = []
    means_2 = []
    sem_1 = []
    sem_2 = []
    for j in range(len(roi_acronyms)):
        data1 = group1_centrality[:, j]
        data2 = group2_centrality[:, j]

        # Ignore NaN values from subjects without S1-i
        data1 = data1[~np.isnan(data1)]  
        data2 = data2[~np.isnan(data2)]
                
        # Test using Mann-Whitney U
        u, p = stats.mannwhitneyu(data1, data2)
        p_values.append(p)

        # Calculate means
        means_1.append(np.mean(data1))
        means_2.append(np.mean(data2))
        
        # Calculate SEM
        sem_1.append(stats.sem(data1))
        sem_2.append(stats.sem(data2))

    # Print centrality results in table format
    print("\nBetweenness Centrality by Region:")
    header = ["ROI", "P-Value", "Mean ± SEM (1)", "Mean ± SEM (2)"]
    table = []
    for region in range(len(roi_acronyms)):
        if p_values[region] >= 0.05:
            continue
        roi_name = roi_acronyms[region]
        p_val = f"{np.round(p_values[region],4)}"
        mean_sem_1 = f"{np.round(means_1[region],3)} ± {np.round(sem_1[region],3)}"
        mean_sem_2 = f"{np.round(means_2[region],3)} ± {np.round(sem_2[region],3)}"
        table.append([roi_name, p_val, mean_sem_1, mean_sem_2])
    print(tabulate(table, headers=header, tablefmt="pretty"))

    # return (
    #     p_values,
    #     means_1,
    #     sem_1,
    #     means_2,
    #     sem_2,
    #     group1_centrality,
    #     group2_centrality,
    # )


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
    fontsize_names=None,
    fontsize_colorbar=None,
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
    label_names = roi_acronyms
    lh_labels = [
        label for label in label_names if label.endswith("lh") or label.endswith("-i")
    ]

    # Get the y-location of the label
    label_ypos_lh = list()
    for name in lh_labels:
        data_idx = label_names.index(name)
        ypos = np.mean(labels[data_idx].pos[:, 1])
        label_ypos_lh.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

    # For the right hemi
    rh_labels = [label[:-2] + "rh" for label in lh_labels]

    # If resting state, change S1-lh and S1-rh to S1-i and S1-c
    if "Eyes" in condition:
        rh_labels[rh_labels.index("S1rh")] = "S1-c"
   
    node_order = lh_labels[::-1] + rh_labels

    # Circular layout
    node_angles = mne.viz.circular_layout(
        label_names,
        node_order,
        start_pos=90,
        group_boundaries=[0, len(label_names) // 2],
    )

    fig, _ = mne_conn.viz.plot_connectivity_circle(
        data,
        roi_acronyms,
        node_edgecolor="white",
        node_angles=node_angles,
        node_colors=node_colors,
        textcolor="black",
        facecolor="white",
        colormap=colormap,
        fontsize_names=fontsize_names,
        fontsize_colorbar=fontsize_colorbar,
        fontsize_title=15,
        vmin=vmin,
        vmax=vmax,
        title=title_prefix,
        fig=fig,
        subplot=subplot,
        show=False,
    )

    # Save figure
    if save_fig:
        # fig.tight_layout()
        filename = f"{condition}_{band}_{method}.png"
        os.makedirs(os.path.join(save_path, "Connectivity circle"), exist_ok=True)
        fig.savefig(
            os.path.join(save_path, "Connectivity circle", filename),
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            dpi=300,
        )


def get_top_connections(data, method, n_top=3):
    """
    Get top n_top connections based on strength of the given FC method

    Parameters
    ----------
    data : 2D array
        Matrix of connectivity values
    n_top : int
        Number of top connections to get

    Returns
    -------
    top_connections : list
        List of tuples of (region pair, strength)
    """
    # Adjust for dpli
    if method == "dpli":
        data = np.abs(data - 0.5)

    # Iterate over the lower triangle of the matrix
    top_connections = []
    for i in range(0, len(data)):
        for j in range(
            0, i + 1
        ):  # Start from 0 to i to exclude the diagonal and upper triangle
            strength = np.round(data[i, j], 3)
            top_connections.append(((i, j), strength))

    # Sort the connections by strength and select the top n_top
    top_connections.sort(key=lambda x: x[1], reverse=True)
    top_connections = top_connections[:n_top]

    return top_connections, strength


def plot_connectivity_and_stats(
    means_1,
    nepochs,
    group_names,
    method,
    band,
    roi_names,
    condition,
    titles,
    save_names,
    save_path,
    group_dict=None,
    means_2=None,
    sem_1=None,
    sem_2=None,
    p_values=None,
    vmin=None,
    vmax=None,
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
    """
    Generate a plot of connectivity and statistics.

    Parameters:
    - means_1: numpy.ndarray, the means for group 1.
    - nepochs: list, the number of epochs for each group.
    - group_names: list, the names of the groups.
    - method: str, the method used for calculating connectivity.
    - band: str, the frequency band used.
    - roi_names: list, the names of the regions of interest.
    - condition: str, the condition used for the plot.
    - titles: list, the titles for each plot.
    - save_names: list, the names for saving the plots.
    - save_path: str, the path to save the plots.
    - means_2: numpy.ndarray, the means for group 2 (optional).
    - sem_1: numpy.ndarray, the standard error of the means for group 1 (optional).
    - sem_2: numpy.ndarray, the standard error of the means for group 2 (optional).
    - p_values: numpy.ndarray, the p-values for the regions of interest (optional).
    - vmin: float, the minimum value for the plot (optional).
    - vmax: float, the maximum value for the plot (optional).
    - fig: matplotlib.figure.Figure, the figure object (optional).
    - subplot: matplotlib.axes._subplots.Subplot, the subplot object (optional).
    - roi_acronyms: list, the acronyms for the regions of interest (optional).
    - save_fig: bool, whether to save the figure (default=True).
    - highlight_pvals: bool, whether to highlight significant p-values (default=True).
    - show_only_significant: bool, whether to show only significant values (default=True).
    - min_fc_val: float, the minimum value to highlight (optional).
    - set_title: bool, whether to set the title (default=True).
    - show_fc_vals: bool, whether to show the functional connectivity values (default=True).
    - round_neg_vals: bool, whether to round negative values (default=False).
    """
    ###############################################################################
    ### FOR PERMUTATION TESTING ###
    sig_pairs = []
    table2 = []
    ###############################

    ### Settings ###
    # Determine whether data provided is individual data or group data
    isindividual = True if means_2 is None else False

    # Set min_fc_val if not provided
    if min_fc_val is None:
        min_fc_val = -1

    # Round negative values in the means
    if round_neg_vals:
        for data in [means_1, means_2] if not isindividual else [means_1]:
            for i in range(len(roi_names)):
                for j in range(len(roi_names)):
                    if data[i, j] < 0:
                        data[i, j] == 0.0

    # Parameters for p-values plot
    if not isindividual:
        # Get highlight indices
        highlight_ij = []
        for i in range(len(roi_names)):
            for j in range(len(roi_names)):
                if p_values[i, j] < 0.05:
                    highlight_ij.append((i, j))

        # Remove any highlights from upper right triangle
        for i in range(len(roi_names)):
            for j in range(i, len(roi_names)):
                # Also remove those from highlight_ij
                if (i, j) in highlight_ij:
                    highlight_ij.remove((i, j))

        # Make top-right diagonal and above white
        for i in range(len(roi_names)):
            for j in range(i, len(roi_names)):
                p_values[i, j] = np.nan

        # If showing only significant values, make the rest appear white
        if show_only_significant:
            for i in range(len(roi_names)):
                for j in range(len(roi_names)):
                    if p_values[i, j] >= 0.05:
                        p_values[i, j] = np.nan

    # Indicate position of p-value plot
    pval_pos = 2

    # Get shortened method name for plot
    method = get_method_plot_name(method)

    # Set font sizes
    overlay_fontsize = 9

    ###############################################################################
    # Print table summary of mean and sem, if not plotting individual data
    if not isindividual:
        # Print the table summary
        print(f"\nMann-Whitney U Test Between {group_names[0]} and {group_names[1]}:")
        header = ["ROI Pair", "P-Value", "Mean ± SEM (1)", "Mean ± SEM (2)"]
        table = []
        for region_pair in highlight_ij:
            roi_pair = (
                f"{roi_acronyms[region_pair[0]]} <-> {roi_acronyms[region_pair[1]]}"
            )
            p_val = np.round(p_values[region_pair[0], region_pair[1]], 4)
            mean_sem_1 = f"{np.round(means_1[region_pair[0], region_pair[1]],3)} ± {np.round(sem_1[region_pair[0], region_pair[1]],3)}"
            mean_sem_2 = f"{np.round(means_2[region_pair[0], region_pair[1]],3)} ± {np.round(sem_2[region_pair[0], region_pair[1]],3)}"

            table.append([roi_pair, p_val, mean_sem_1, mean_sem_2])
            table2.append([roi_acronyms[region_pair[0]], roi_acronyms[region_pair[1]], p_val, mean_sem_1, mean_sem_2])
        print(tabulate(table, headers=header, tablefmt="pretty"))
        sig_pairs = table2
    else:
        print(f"Top 3 Connections in {group_names} group")
        header = ["Top connections", f"{method} Value"]
        table = []

        top_connections, strength = get_top_connections(means_1, method, n_top=3)

        for region_pair, strength in top_connections:
            roi_pair = (
                f"{roi_acronyms[region_pair[0]]} <-> {roi_acronyms[region_pair[1]]}"
            )
            method_val = np.round(strength, 3)

            table.append([roi_pair, method_val])
        print(tabulate(table, headers=header, tablefmt="pretty"))

    # Choose the colormap
    colormap = "hot_r"

    # Loop through means and p values for plotting
    for (
        data_idx,
        data,
    ) in zip(
        range(len(titles)),
        [
            means_1,
            means_2,
            p_values,
        ],
    ):

        # # Plot parameters
        if vmin is None or vmax is None:  # if not already set
            if data_idx == pval_pos:
                vmin, vmax = (0.0, 1.0)
            else:
                if method == "dwPLI":
                    vzero = 0.0
                    vtolerance = 0.25
                    vmin, vmax = (vzero, vzero + vtolerance)
                elif method == "dPLI":
                    vzero = 0.5
                    vtolerance = 0.2
                    vmin, vmax = (vzero - vtolerance, vzero + vtolerance)
                elif "Pairwise" in method:
                    vzero = 0.0
                    vtolerance = 0.7
                    vmin, vmax = (vzero, vzero + vtolerance)
                elif "Symmetric" in method:
                    vzero = 0.0
                    vtolerance = 0.3
                    vmin, vmax = (vzero, vzero + vtolerance)
                else:
                    print(
                        f"Method {method} not supported for vmin and vmax calculation."
                    )

        else:
            # set from arguments + the preset for p-value plot
            if data_idx == pval_pos:
                vmin, vmax = (0.0, 1.0)

        # Plot circle for FC values, and connectivity matrix just for p-values
        if data_idx == pval_pos and not isindividual:
            fig = plt.figure()
            plt.imshow(data, vmin=vmin, vmax=vmax, cmap="hot")

            plt.ylabel("Regions", labelpad=20)
            plt.yticks(range(len(roi_acronyms)), labels=roi_acronyms)

            plt.xlabel("Regions", labelpad=20)
            plt.xticks(
                range(len(roi_acronyms)), labels=roi_acronyms, rotation=45, ha="right"
            )

            if set_title:
                if data_idx != pval_pos:  # group 1 or group 2
                    plt.title(
                        f"{titles[data_idx]} | {condition} | {band} | ({method} method, {nepochs[data_idx-1]} trials)"
                    )
                else:  # p-values
                    plt.title(
                        f"{titles[data_idx]} | {condition} | {band} | ({method} method, {nepochs[0]} vs. {nepochs[1]} trials)"
                    )

        else:
            # First change vmin and vmax for individual plots
            if isindividual:
                if method == "dwPLI":
                    vzero = 0.0
                    vtolerance = 1.0
                    vmin, vmax = (vzero, vzero + vtolerance)
                elif method == "dPLI":
                    vzero = 0.5
                    vtolerance = 0.4
                    vmin, vmax = (vzero - vtolerance, vzero + vtolerance)
                elif "Pairwise" in method:
                    vzero = 0.0
                    vtolerance = 1.0
                    vmin, vmax = (vzero, vzero + vtolerance)
                elif "Symmetric" in method:
                    vzero = 0.0
                    vtolerance = 1.0
                    vmin, vmax = (vzero, vzero + vtolerance)

            plt.figure()
            plot_connectivity_circle(
                data=data,
                method=method,
                band=band,
                roi_names=roi_names,
                roi_acronyms=roi_acronyms,
                condition=condition,
                save_path=save_path,
                colormap=colormap,
                vmin=vmin,
                vmax=vmax,
                fontsize_names=13,
                fontsize_colorbar=13,
                title_prefix=f"{titles[data_idx]}",
                save_fig=True,
            )

        # Overlay values
        if data_idx == pval_pos:  # if plotting matrix
            for i in range(len(roi_names)):
                for j in range(len(roi_names)):
                    if data[i, j] < 0.05 and not np.isnan(data[i, j]):
                        if show_fc_vals:
                            plt.text(
                                j,
                                i,
                                round(data[i, j], 3),
                                ha="center",
                                va="center",
                                color="w",
                                fontsize=overlay_fontsize,
                            )

        # Add rectangles for highlighted squares
        if highlight_pvals:
            for i, j in highlight_ij:
                plt.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )

        filename = f"{condition}_{band}_{method}.png"
        if fig and save_fig and not isindividual:
            fig.savefig(os.path.join(save_path, filename), bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()
    return sig_pairs