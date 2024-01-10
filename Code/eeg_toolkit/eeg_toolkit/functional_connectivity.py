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
    label_ts,
    roi_names,
    method,
    Freq_Bands,
    tmin,
    tmax,
    sfreq=400,
    plot_freq_matrix=False,
):
    if plot_freq_matrix is True:
        fmin, fmax = 4, 100
        # Compute connectivity over trials
        con_epochs = mne_conn.spectral_connectivity_epochs(
            label_ts,
            method=method,
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            mt_adaptive=True,
            n_jobs=-1,
        )
        plot_con_freqx(con_epochs, roi_names)
    else:
        fmins = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
        fmaxs = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])

        for fmin, fmax in zip(fmins, fmaxs):
            con_epochs = mne_conn.spectral_connectivity_epochs(
                label_ts,
                method=method,
                mode="multitaper",
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                mt_adaptive=True,
                n_jobs=1,
            )
    print(con_epochs.shape)
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


def compute_sub_avg_con(
    sub_ids_CP,
    sub_ids_HC,
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
):
    """
    Compute the average connectivity for each subject in two groups.

    Parameters:
    - sub_ids_CP: A list of subject IDs in the CP group.
    - sub_ids_HC: A list of subject IDs in the HC group.
    - processed_data_path: The path to the processed data.
    - zscored_epochs_data_path: The path to the z-scored epochs data.
    - EO_resting_data_path: The path to the resting state data with eyes open.
    - EC_resting_data_path: The path to the resting state data with eyes closed.
    - connectivity_methods: A list of connectivity methods to compute.
    - roi_names: A list of ROI names.
    - Freq_Bands: A list of frequency bands.
    - tmin: The starting time of the epochs.
    - tmax: The ending time of the epochs.
    - sfreq: The sampling frequency.

    Returns:
    - results: A dictionary containing the computed connectivity results for each subject and condition.
    """
    # Initialize dictionary to store results
    results = {}

    for group in [tuple(sub_ids_CP), tuple(sub_ids_HC)]:
        # initialize dictionary for this group
        results[group] = {}
        for sub_id in sub_ids_CP:
            # Initialize dictionary for this subject
            results[group][sub_id] = {}

            # First load in STC data and separate epochs by stimulus
            # Epochs
            (
                hand_all_label_ts,
                back_all_label_ts,
                hand_all_ratings,
                back_all_ratings,
            ) = separate_epochs_by_stim(
                sub_id, processed_data_path, zscored_epochs_data_path
            )
            # Resting state
            # label_ts_EO = utils.unpickle_data(
            #     EO_resting_data_path, f"{sub_id}_EO.pkl"
            # )
            # label_ts_EC = utils.unpickle_data(
            #     EC_resting_data_path, f"{sub_id}_EC.pkl"
            # )

            # Unpack label_ts for each site and stimulus level
            label_ts_all = []
            label_ts_all.extend(list(hand_all_label_ts))
            label_ts_all.extend(list(back_all_label_ts))
            # label_ts_all.extend([label_ts_EO, label_ts_EC])

            # Compute connectivity for epochs
            for method in connectivity_methods:
                for label_ts, condition in zip(label_ts_all, conditions):
                    if isinstance(label_ts, list):
                        con_epochs = compute_connectivity_epochs(
                            label_ts, roi_names, method, Freq_Bands, tmin, tmax, sfreq
                        )

                        # average epochs within subject first
                        con_epochs_mean = np.mean(con_epochs.get_data(), axis=1)

                        # Add result to dictionary
                        if condition not in results[group][sub_id]:
                            results[group][sub_id][condition] = {}
                        results[group][sub_id][condition][method] = con_epochs_mean
                    else:
                        con_eo = compute_connectivity_resting_state(
                            label_ts,
                            roi_names,
                            method,
                            Freq_Bands,
                            sfreq,
                            condition="EO",
                        )
                        con_ec = compute_connectivity_resting_state(
                            label_ts,
                            roi_names,
                            method,
                            Freq_Bands,
                            sfreq,
                            condition="EC",
                        )

                        # Add result to dictionary
                        if "Eyes Open" not in results[group][sub_id]:
                            results[group][sub_id]["Eyes Open"] = {}
                        results[group][sub_id]["Eyes Open"][method] = con_eo

                        if "Eyes Closed" not in results[group][sub_id]:
                            results[group][sub_id]["Eyes Closed"] = {}
                        results[group][sub_id]["Eyes Closed"][method] = con_ec

    return results


def compute_group_avg_con(results, fc_path, Freq_Bands, roi_names):
    """
    Computes the average results for each group, condition, and method based on the given results.

    Args:
        results (dict): A dictionary containing the results for each group, condition, method, and subject.
        fc_path (str): The path to the file containing the frequency bands.
        Freq_Bands (list): A list of frequency bands.
        roi_names (list): A list of ROI names.

    Returns:
        dict: A dictionary containing the average results for each group, condition, and method.
    """
    # Initialize dictionary to store average results
    avg_results = {}

    # Loop through each group
    for group in results.keys():
        avg_results[group] = {}

        # Loop through each condition
        for condition in results[group][next(iter(results[group]))].keys():
            avg_results[group][condition] = {}

            # Loop through each method
            for method in results[group][next(iter(results[group]))][condition].keys():
                # Initialize list to store all subjects' data for this group, condition, and method
                all_subjects_data = []

                # Loop through each subject
                for sub_id in results[group].keys():
                    # Append this subject's data to the list
                    all_subjects_data.append(results[group][sub_id][condition][method])

                # Convert list to numpy array
                all_subjects_data = np.array(all_subjects_data)

                # Compute mean across all subjects and store in avg_results
                avg_results[group][condition][method] = np.mean(
                    all_subjects_data, axis=0
                )

                avg_result_current = avg_results[group][condition][method]
                print(avg_result_current)
                # Plot averaged data and save
    return avg_results


def plot_con_freqx(con_epochs, roi_names):
    freqs = con_epochs.freqs
    fmin, fmax = 4, 100
    n_rows, n_cols = con_epochs.get_data(output="dense").shape[:2]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 20), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n_rows):
        for j in range(i + 1):
            if i == j:
                axes[i, j].set_axis_off()
                continue

            # get data
            con_data = con_epochs.get_data(output="dense")[i, j, :]
            axes[i, j].plot(freqs, con_data)
            axes[j, i].plot(freqs, con_data)

            if j == 0:
                axes[i, j].set_ylabel(roi_names[i])
                axes[0, i].set_title(roi_names[i])
            if i == (n_rows - 1):
                axes[i, j].set_xlabel(roi_names[j])
            axes[i, j].set(xlim=[fmin, fmax], ylim=[-0.2, 1])
            axes[j, i].set(xlim=[fmin, fmax], ylim=[-0.2, 1])

            # Show band limits
            for f in [8, 13, 30, 100]:
                axes[i, j].axvline(f, color="k")
                axes[j, i].axvline(f, color="k")
            # Show line-noise
            axes[j, i].axvspan(
                58.5,
                61.5,
                color="k",
                alpha=0.3,
            )
    plt.tight_layout()
    plt.show()


def plot_and_save(
    avg_con, plot_func, method, Freq_Bands, roi_names, group, condition, output_dir
):
    """
    Creates and saves plots for each combination of group, condition, and method in avg_results.

    Parameters:
    - avg_con: The  averaged connectivity matrix.
    - plot_func: The function to use for creating the plot.
    - output_dir: The directory where the plots should be saved.
    """

    # Create a new figure for this plot
    plt.figure()

    # Create the plot using the provided function
    plot_func(avg_con, method, Freq_Bands, roi_names)

    # Set the title of the plot to indicate the group, condition, and method
    plt.title(f"Group: {group}, Condition: {condition}, Method: {method}")

    # Save the figure
    # The filename is created from the group, condition, and method to ensure it's unique
    filename = f"{func_name}_{group}_{condition}_{method}.png"
    plt.savefig(os.path.join(output_dir, filename))

    # Close the figure to free up memory
    plt.close()


def plot_con_matrix(
    con_data, n_connectivity_methods, connectivity_methods, roi_names, foi
):
    """Visualize the connectivity matrix."""
    fig, ax = plt.subplots(
        1, n_connectivity_methods, figsize=(6 * n_connectivity_methods, 6)
    )
    for c in range(n_connectivity_methods):
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


def plot_connectivity(con_epochs, t_con_max, roi_names, connectivity_methods):
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

        ax.set_title(f"{connectivity_methods[c]} Connectivity")

        plt.show()


def plot_connectivity_circle(con_matrices, connectivity_methods, Freq_Bands, roi_names):
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
