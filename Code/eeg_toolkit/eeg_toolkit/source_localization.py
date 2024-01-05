##################################################################
from .utils import clear_display, set_montage
import mne
import os
import numpy as np
from mne.datasets import fetch_fsaverage

mne.set_log_level("WARNING")

#################################################################################################

# Source the fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subject = "fsaverage"
trans = "fsaverage"
subjects_dir = os.path.dirname(fs_dir)
src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")  # surface for dSPM
bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
model_fname = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem.fif")
snr = 3.0  # for inverse
#################################################################################################


class InstanceNotDefinedError(Exception):
    pass


def load_raw(data_path, sub_id, condition):
    """
    Load a raw data file and preprocess it.
    """
    sub_fname = f"{sub_id}_{condition}-raw.fif"
    raw_path = os.path.join(data_path, sub_fname)
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw.set_eeg_reference("average", projection=True)
    return raw


def make_sub_time_win_path(sub_id, save_path):
    """
    Make a subject's time window data path.
    """
    sub_path = os.path.join(save_path, sub_id)
    os.makedirs(sub_path, exist_ok=True)
    return sub_path


def zscore_epochs(sub_id, data_path, tmin, raw_eo):
    epochs_fname = f"{sub_id}_preprocessed-epo.fif"
    epochs = mne.read_epochs(os.path.join(data_path, epochs_fname))
    epochs.set_eeg_reference("average", projection=True)

    data_epochs = epochs.get_data()
    data_zepo = np.zeros_like(data_epochs)
    base_data = epochs.get_data(tmin=tmin, tmax=0.0)

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
        on_missing="ignore",
        event_id=epochs.event_id,
        events=epochs.events,
    )
    set_montage(zepochs, raw_eo.get_montage())

    return zepochs


def save_label_time_course(
    sub_id,
    condition,
    snr,
    trans,
    src,
    bem,
    mne_object,
    noise_cov,
    labels,
    save_path,
    average_dipoles=True,
):
    apply_inverse_kwargs = dict(
        lambda2=1.0 / snr**2,
        verbose=True,
    )

    fwd = mne.make_forward_solution(
        mne_object.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        n_jobs=-1,
        verbose=True,
    )
    clear_display()

    inverse_operator = mne.minimum_norm.make_inverse_operator(
        mne_object.info, fwd, noise_cov, verbose=True
    )

    def apply_inverse_Raw(
        mne_object,
        inverse_operator,
        save_path,
        sub_id,
        condition,
        average_dipoles=False,
    ):
        sub_id_if_nan = None
        stc = mne.minimum_norm.apply_inverse_raw(
            mne_object, inverse_operator, method="dSPM", **apply_inverse_kwargs
        )
        src = inverse_operator["src"]
        print(f"Saving {sub_id} {condition}")
        mode = "mean_flip" if average_dipoles else None
        label_ts = mne.extract_label_time_course(stc, labels, src, mode=mode)
        if np.isnan(label_ts).any():
            sub_id_if_nan = sub_id
            # raise ValueError("label_ts contains nan")
        stc_mean_flip = mne.labels_to_stc(labels, label_ts, src=src)
        stc_mean_flip.save(
            os.path.join(save_path, f"{sub_id}_{condition}.stc"), overwrite=True
        )
        return label_ts, sub_id_if_nan

    def apply_inverse_Epochs(
        mne_object,
        inverse_operator,
        save_path,
        sub_id,
        condition,
        average_dipoles=False,
    ):
        sub_id_if_nan = None
        stc = mne.minimum_norm.apply_inverse_epochs(
            mne_object, inverse_operator, method="dSPM", **apply_inverse_kwargs
        )
        src = inverse_operator["src"]
        print(f"Saving {sub_id} {condition}")
        mode = "mean_flip" if average_dipoles else None
        label_ts = mne.extract_label_time_course(stc, labels, src, mode=mode)
        if np.isnan(label_ts).any():
            sub_id_if_nan = sub_id
            # raise ValueError("label_ts contains nan")

        for epoch_idx, epoch_ts in enumerate(label_ts):
            print(f"Saving STC for epoch {epoch_idx}")
            stc_mean_flip = mne.labels_to_stc(labels, epoch_ts, src=src)
            stc_mean_flip.save(
                os.path.join(save_path, f"{sub_id}_{condition}_{epoch_idx}.stc"),
                overwrite=True,
            )
        return label_ts, sub_id_if_nan

    if isinstance(mne_object, mne.io.fiff.raw.Raw):
        print("Applying inverse to Raw object")
        label_ts, sub_id_if_nan = apply_inverse_Raw(
            mne_object, inverse_operator, save_path, sub_id, condition, average_dipoles
        )
    elif isinstance(mne_object, mne.epochs.EpochsArray):
        print("Applying inverse to Epochs object")
        label_ts, sub_id_if_nan = apply_inverse_Epochs(
            mne_object, inverse_operator, save_path, sub_id, condition, average_dipoles
        )
    else:
        raise ValueError("Invalid mne_object type")
    return label_ts, sub_id_if_nan
    clear_display()


def to_source(
    sub_id,
    data_path,
    zscored_epochs_save_path,
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
        zscored_epochs_save_path (str): The path to save the Z-scored epochs.
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

    #################################################################################################

    # Convert ROI names to labels
    labels = [
        mne.read_labels_from_annot(subject, regexp=roi, subjects_dir=subjects_dir)[0]
        for roi in roi_names
    ]
    roi_names_count = len(roi_names)

    # Extract time window information from tuple arguments
    tmin, tmax, bmax = times_tup
    # TODO: remove hardcoded noise window once generated for each subject
    rest_min, rest_max = 5.5, 7.5

    # Compute noise & data covariance
    # TODO: replace below with actual noise segment
    # noise_segment = load_raw(data_path, "noise", sub_id)
    # raw_eo = load_raw(data_path, sub_id, "EO")
    raw_eo = load_raw(data_path, sub_id, "preprocessed")
    noise_segment = raw_eo.crop(tmin=rest_min * 60, tmax=rest_max * 60)
    noise_cov = mne.compute_raw_covariance(noise_segment, verbose=True)
    # Regularize the covariance matrices
    noise_cov = mne.cov.regularize(noise_cov, noise_segment.info, eeg=0.1, verbose=True)

    #################################################################################################

    # If processing resting, check directories for count
    if return_EO_resting:
        EO_subpath = make_sub_time_win_path(sub_id, EO_resting_save_path)
        EO_subpath_count = len(os.listdir(EO_subpath))
    if return_EC_resting:
        raw_ec = load_raw(data_path, sub_id, "EC")
        EC_subpath = make_sub_time_win_path(sub_id, EC_resting_save_path)
        EC_subpath_count = len(os.listdir(EC_subpath))

    # If processing epochs, check directory for count
    if return_zepochs:
        Zepo_subpath = make_sub_time_win_path(sub_id, zscored_epochs_save_path)
        Zepo_subpath_count = len(os.listdir(Zepo_subpath))

    #################################################################################################

    # If desired and eyes open resting data not yet processed, process it
    label_ts_EO, label_ts_EC, label_ts_Epochs = None, None, None
    sub_id_if_nan = None
    if return_EO_resting and EO_subpath_count < roi_names_count:
        label_ts_EO, sub_id_if_nan = save_label_time_course(
            sub_id,
            "EO",
            snr,
            trans,
            src,
            bem,
            raw_eo,
            noise_cov,
            labels,
            EO_subpath,
            average_dipoles=True,
        )

    # If desired and eyes closed resting data not yet processed, process it
    if return_EC_resting and EC_subpath_count < roi_names_count:
        label_ts_EC, sub_id_if_nan = save_label_time_course(
            sub_id,
            "EC",
            snr,
            trans,
            src,
            bem,
            raw_ec,
            noise_cov,
            labels,
            EC_subpath,
            average_dipoles=True,
        )

    #################################################################################################

    # If desired and epochs not yet processed, Z-score and source localize
    if return_zepochs and Zepo_subpath_count < roi_names_count:
        zepochs = zscore_epochs(sub_id, data_path, tmin, raw_eo)

        label_ts_Epochs, sub_id_if_nan = save_label_time_course(
            sub_id,
            "epochs",
            snr,
            trans,
            src,
            bem,
            zepochs,
            noise_cov,
            labels,
            Zepo_subpath,
            average_dipoles=True,
        )

    return (label_ts_EO, label_ts_EC, label_ts_Epochs), sub_id_if_nan
