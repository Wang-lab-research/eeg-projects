##################################################################
from .utils import *
from mne.datasets import fetch_fsaverage

mne.set_log_level("WARNING")

# Source the fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subject = "fsaverage"
trans = "fsaverage"
subjects_dir = os.path.dirname(fs_dir)
src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")  # surface for dSPM
bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
model_fname = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem.fif")

# apply inverse
snr = 3.0
apply_inverse_rawEC_kwargs = dict(
    lambda2=1.0 / snr**2, verbose=True  # regularizer parameter (λ²)
)
##################################################################


def load_raw(data_path, sub_id, condition):
    """
    Load a raw data file and preprocess it.

    Parameters:
        data_path (str): The path to the directory containing the data files.
        sub_id (str): The subject ID.
        condition (str): The condition of the data.

    Returns:
        raw (mne.io.Raw): The preprocessed raw data.

    Raises:
        FileNotFoundError: If the raw data file does not exist.

    Notes:
        - This function assumes that the raw data file follows the naming convention: "{sub_id}_{condition}-raw.fif".
        - The raw data file is loaded using the MNE library.
        - The loaded raw data is preprocessed by setting an EEG reference to "average" with projection.
    """
    sub_fname = f"{sub_id}_{condition}-raw.fif"
    raw_path = os.path.join(data_path, sub_fname)
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    print(sub_raw_fname)
    raw.set_eeg_reference("average", projection=True)
    return raw


def make_sub_time_win_path(sub_id, save_path):
    """
    Make a subject's time window data path.
    """
    sub_path = os.path.join(save_path, sub_id)
    [os.mkdir(path) for path in [sub_path] if not os.path.exists(path)]

    return sub_path


def to_source(
    sub_id,
    data_path,
    Z_scored_epochs_save_path,
    EC_resting_save_path,
    EO_resting_save_path,
    roi_names,
    times_tup,
    return_zepochs=True,
    return_EC_resting=True,
    return_EO_resting=True,
    average_dipoles=True,
):
    """
    Compute the source localization for a subject for eyes closed, eyes open, and z-scored epochs.
    Args:
        sub_id (str): The ID of the subject.
        data_path (str): The path to the data.
        Z_scored_epochs_save_path (str): The path to save the Z-scored epochs.
        EC_resting_save_path (str): The path to save the EC resting state data.
        EO_resting_save_path (str): The path to save the EO resting state data.
        roi_names (list): The names of the ROIs.
        times_tup (tuple): A tuple containing the time window information.
        return_zepochs (bool, optional): Whether to return the Z-scored epochs. Defaults to True.
        return_EC_resting (bool, optional): Whether to return the EC resting state data. Defaults to True.
        return_EO_resting (bool, optional): Whether to return the EO resting state data. Defaults to True.
        average_dipoles (bool, optional): Whether to average the dipoles. Defaults to True.

    Returns:
        None
    """

    # Load the resting state data
    rawEO = load_raw(data_path, "EO", sub_id)
    rawEC = load_raw(data_path, "EC", sub_id)
    # TODO: uncomment when noise segment is generated
    noise_segment = load_raw(data_path, "noise", sub_id)

    labels = [
        mne.read_labels_from_annot(subject, regexp=roi, subjects_dir=subjects_dir)[0]
        for roi in roi_names
    ]

    # Extract time window information from tuple arguments
    tmin, tmax, bmax = times_tup
    # TODO: remove hardcoded noise window once generated for each subject
    rest_min, rest_max = 5.5, 7.5

    # Make subpaths
    if return_EC_resting:
        EC_subpath = make_sub_time_win_path(sub_id, EC_resting_save_path)
        EC_subpath_count = len(os.listdir(EC_subpath))
    if return_EO_resting:
        EO_subpath = make_sub_time_win_path(sub_id, EO_resting_save_path)
        EO_subpath_count = len(os.listdir(EO_subpath))
    if return_zepochs:
        Zepo_subpath = make_sub_time_win_path(sub_id, Z_scored_epochs_save_path)
        Zepo_subpath_count = len(os.listdir(Zepo_subpath))
    roi_names_count = len(roi_names)

    # Quick check to see if subject already processed
    if return EO_subpath_count < roi_names_count:
        # Z-score Epochs then convert to STC
        if return_zepochs and Zepo_subpath_count < roi_names_count:
            sub_epo_fname = f"{sub_id}_preprocessed-epo.fif"
            print(sub_epo_fname)
            epochs = mne.read_epochs(os.path.join(epo_path, sub_epo_fname))

            epochs.set_eeg_reference("average", projection=True)

            data_epo = epochs.get_data()
            data_zepo = np.zeros_like(data_epo)
            base_data = epochs.get_data(tmin=tmin, tmax=bmax)

            for epoch_idx in range(data_epochs.shape[0]):
                for channel_idx in range(data_epochs.shape[1]):
                    base_mean = np.mean(base_data[epoch_idx, channel_idx, :])
                    base_std = np.std(base_data[epoch_idx, channel_idx, :])
                    data_zepo[epoch_idx, channel_idx, :] = (
                        data_epochs[epoch_idx, channel_idx, :] - base_mean
                    ) / base_std

            zepochs = mne.EpochsArray(
                data_zepo,
                info=epochs.info,
                tmin=tmin,
                on_missing="ignore",  # ignore missing Hand LS and Back LS
                event_id=epochs.event_id,
                events=epochs.events,
            )
            set_montage(zepochs, rawEC.get_montage())

        # Crop resting EC and EO
        # TODO: perform crop for resting state
        # TODO: use different noise covariance for resting state

        # Compute noise & data covariance
        rawEC_crop = rawEC.copy().crop(tmin=60 * rest_min, tmax=60 * rest_max)
        noise_cov = mne.compute_rawEC_covariance(rawEC_crop, verbose=True)

        ################################### Regularize the covariance matrices ##########################################
        noise_cov = mne.cov.regularize(
            noise_cov, rawEC_crop.info, eeg=0.1, verbose=True
        )

        #################################### Compute the forward solution ###############################################
        fwd = mne.make_forward_solution(
            rawEC.info,
            trans=trans,
            src=src,
            bem=bem,
            meg=False,
            eeg=True,
            n_jobs=-1,
            verbose=True,
        )
        clear_display()

        ###################################### Make the inverse operator ###############################################
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            rawEC.info, fwd, noise_cov, verbose=True
        )

        if return_zepochs:
            inverse_operator_zepo = mne.minimum_norm.make_inverse_operator(
                zepochs.info, fwd, noise_cov, verbose=True
            )
        clear_display()

        ################################# Save source time courses #######################################
        if len(os.listdir(EO_subpath)) < len(roi_names):
            print(sub_rawEC_fname)
            src_cont = inverse_operator["src"]
            stc_cont = mne.minimum_norm.apply_inverse_rawEC(
                rawEC, inverse_operator, method="dSPM", **apply_inverse_rawEC_kwargs
            )
            # Save the continuous STC file
            print(f"Saving {sub_id} continuous stc")
            if not average_dipoles:
                label_ts = mne.extract_label_time_course(
                    stc_cont, labels, src_cont, return_generator=True
                )
                label_ts.save(os.path.join(save_path_cont, sub_id, overwrite=True))
            elif average_dipoles:
                label_ts = mne.extract_label_time_course(
                    stc_cont, labels, src_cont, mode="mean_flip", return_generator=True
                )
                label_ts.save(os.path.join(save_path_cont, sub_id, overwrite=True))
            clear_display()

        ################################# Apply inverse to epochs #######################################
        if return_zepochs:
            if len(os.listdir(Zepo_subpath)) < len(roi_names):
                print(sub_epo_fname)
                zepochs_stc = mne.minimum_norm.apply_inverse_epochs(
                    zepochs,
                    inverse_operator_zepo,
                    method="dSPM",
                    **apply_inverse_rawEC_kwargs,
                )

                src_zepo = inverse_operator_zepo["src"]
                if not average_dipoles:
                    label_ts = mne.extract_label_time_course(
                        stc_zepo, labels, src_zepo, return_generator=True
                    )
                    label_ts.save(os.path.join(save_path_zepo, sub_id, overwrite=True))
                elif average_dipoles:
                    label_ts = mne.extract_label_time_course(
                        stc_zepo,
                        labels,
                        src_zepo,
                        mode="mean_flip",
                        return_generator=True,
                    )
                    label_ts.save(os.path.join(save_path_zepo, sub_id, overwrite=True))

                print(f"Saving {sub_id} zepochs stc")
                clear_display()

    # return zepochs_stc_arr
