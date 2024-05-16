import mne
import os
import numpy as np
from mne.datasets import fetch_fsaverage
import sys

# import hdf5storage
from scipy.io import savemat

sys.path.append("/home/wanglab/Documents/George Kenefati/Code/eeg_toolkit/")
from eeg_toolkit import utils  # noqa: E402

mne.set_log_level("WARNING")

# Source the fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subject = "fsaverage"
trans = "fsaverage"
subjects_dir = os.path.dirname(fs_dir)
src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")  # surface for dSPM
bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
model_fname = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem.fif")
snr = 1.0  # for non-averaged data


def load_raw(processed_data_path, sub_id, condition):
    """
    Load a raw data file and preprocess it.
    """
    sub_fname = f"{sub_id}_{condition}-raw.fif"
    raw_path = os.path.join(processed_data_path, sub_fname)
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw.set_eeg_reference("average", projection=True)
    return raw


def zscore_epochs(sub_id, processed_data_path, tmin, raw):
    """
    Calculate the z-scores for each epoch in the given EEG dataset.

    Args:
        sub_id (str): The subject ID.
        processed_data_path (str): The path to the data directory.
        tmin (float): The start time of the epochs in seconds.
        raw (mne.Raw): The raw EEG data.

    Returns:
        zepochs (mne.EpochsArray): The z-scored epochs.

    Raises:
        FileNotFoundError: If the epochs file is not found.

    """
    epochs_fname = f"{sub_id}_preprocessed-epo.fif"
    epochs = mne.read_epochs(os.path.join(processed_data_path, epochs_fname))
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
    utils.set_montage(zepochs, raw.get_montage())

    return zepochs


def apply_inverse_and_save(
    mne_object,
    inverse_operator,
    labels,
    save_path,
    save_fname,
    sub_id,
    condition,
    method,
    average_dipoles=True,
    save_stc_mat=False,
):
    """
    Apply inverse operator to MNE object and save STC files.

    Args:
        mne_object (object): The MNE object to apply inverse operator to.
        inverse_operator (object): The inverse operator to be used.
        labels (list): The labels to extract time courses from.
        save_path (str): The path to save the STC files.
        sub_id (str): The subject ID.
        condition (str): The condition.
        average_dipoles (bool, optional): Whether to average dipoles. Defaults to True.

    Returns:
        tuple: A tuple containing the label time courses and the subject ID if label time courses contain NaN values.
    """
    apply_inverse_and_save_kwargs = dict(
        lambda2=1.0 / snr**2,
        verbose=True,
    )
    sub_id_if_nan = None
    label_ts = None
    if isinstance(mne_object, mne.io.fiff.raw.Raw):
        print("Applying inverse to Raw object")
        stc = mne.minimum_norm.apply_inverse_raw(
            mne_object, inverse_operator, method=method, **apply_inverse_and_save_kwargs
        )
    elif isinstance(mne_object, mne.epochs.EpochsArray):
        print("Applying inverse to Epochs object")
        stc = mne.minimum_norm.apply_inverse_epochs(
            mne_object, inverse_operator, method=method, **apply_inverse_and_save_kwargs
        )
    else:
        raise ValueError("Invalid mne_object type")

    # Extract labels and do mean flip
    src = inverse_operator["src"]
    mode = "mean_flip" if average_dipoles else None
    label_ts = mne.extract_label_time_course(stc, labels, src, mode=mode)

    # Save as pickle
    if not save_stc_mat:
        utils.pickle_data(save_path, save_fname, label_ts)

    # Save Z-scored Epochs STC only. MAT file for analysis in MATLAB
    elif save_stc_mat and isinstance(label_ts, list):
        # Reshape for convention (optional)
        nepochs = len(label_ts)
        label_ts = np.concatenate(label_ts)
        label_ts = np.reshape(label_ts, (nepochs, len(labels), len(mne_object.times)))
        print("*label_ts shape = ", label_ts.shape)

        for i in range(len(labels)):
            print(f"Saving stc.mat for {sub_id} in region: {labels[i].name}")
            label_ts_i = label_ts[:, i, :]
            print("*label_ts_i shape = ", label_ts_i.shape)

            # Save STC Zepochs per region
            matfiledata = {"data": label_ts_i}
            save_fname = f"{labels[i].name}_{condition}.mat"
            # hdf5storage.write(
            #     matfiledata,
            #     filename=os.path.join(sub_save_path, save_fname),
            #     matlab_compatible=True,
            # )
            sub_save_path = os.path.join(save_path, sub_id)
            savemat(os.path.join(sub_save_path, save_fname), matfiledata)

    # Save subject ID if label time courses contain NaN values
    if np.isnan(label_ts).any():
        sub_id_if_nan = sub_id
        # raise ValueError("label_ts contains nan")

    utils.clear_display()

    return label_ts, sub_id_if_nan


def compute_fwd_and_inv(
    sub_id,
    condition,
    snr,
    trans,
    src,
    bem,
    mne_object,
    noise_var,
    labels,
    save_path,
    save_fname,
    method,
    average_dipoles=True,
    save_stc_mat=False,
    save_inv=True,
):
    """
    Save the time course data for specified labels.

    Parameters:
        sub_id (str): The subject ID.
        condition (str): The condition of the data.
        snr (float): The signal-to-noise ratio.
        trans (str): The path to the transformation matrix file.
        src (str): The path to the source space file.
        bem (str): The path to the BEM model file.
        mne_object (MNE object): The MNE object containing the data.
        noise_var (MNE object): The noise covariance matrix.
        labels (list of str): The labels to save the time course for.
        save_path (str): The path to save the time course data.
        average_dipoles (bool, optional): Whether to average dipoles (default: True).

    Returns:
        None
    """
    # Check if files already saved. Check already in place for regular save to pkl
    sub_done = False
    if save_stc_mat:
        sub_save_path = os.path.join(save_path, sub_id)
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        if len(os.listdir(sub_save_path)) >= len(labels):
            sub_done = True

    label_ts, sub_id_if_nan = None, None  # Initialize variables
    if not sub_done:
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
        utils.clear_display()

        inverse_operator = mne.minimum_norm.make_inverse_operator(
            mne_object.info, fwd, noise_var, verbose=True
        )

        label_ts, sub_id_if_nan = apply_inverse_and_save(
            mne_object,
            inverse_operator,
            labels,
            save_path,
            save_fname,
            sub_id,
            condition,
            method=method,
            average_dipoles=True,
            save_stc_mat=save_stc_mat,
        )

        # Save inverse operator for just one of the data types, make it the epochs
        if save_inv and isinstance(label_ts, list):
            utils.pickle_data(save_path, f"{sub_id}_inv.pkl", inverse_operator)

    return label_ts, sub_id_if_nan
    utils.clear_display()


def to_source(
    sub_id,
    processed_data_path,
    zscored_epochs_save_path,
    EC_resting_save_path,
    EO_resting_save_path,
    roi_names,
    times_tup,
    method,
    return_zepochs=True,
    return_EC_resting=False,
    return_EO_resting=False,
    average_dipoles=True,
    save_stc_mat=False,
    save_inv=True,
):
    """
    Compute the source localization for a subject for eyes closed, eyes open, and z-scored epochs.
    Args:
        sub_id (str): The ID of the subject.
        processed_data_path (str): The path to the data.
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

    # Convert ROI names to labels
    labels = [
        mne.read_labels_from_annot(subject, regexp=roi, subjects_dir=subjects_dir)[0]
        for roi in roi_names
    ]

    # Extract time window information from tuple arguments
    tmin, tmax, bmax = times_tup

    # Compute noise & data covariance
    eo_segment = load_raw(processed_data_path, sub_id, condition="eyes_open")
    noise_cov = mne.compute_raw_covariance(eo_segment, verbose=True)
    # Regularize the covariance matrices
    noise_cov = mne.cov.regularize(noise_cov, eo_segment.info, eeg=0.1, verbose=True)

    # Extract the diagonal elements
    noise_var = mne.Covariance(
        data=np.diag(np.diag(noise_cov.data)),
        names=eo_segment.info["ch_names"],
        bads=eo_segment.info["bads"],
        projs=eo_segment.info["projs"],
        nfree=eo_segment.info["nchan"],
        verbose=True,
    )

    #################################################################################################

    # TODO: Control regions only?
    control_regions = True
    
    # If processing resting, check directories for count
    raw = load_raw(processed_data_path, sub_id, condition="preprocessed")
    if return_EO_resting:
        raw_eo = load_raw(processed_data_path, sub_id, condition="eyes_open")
        EO_save_fname = f"{sub_id}_eyes_open.pkl" if not control_regions else f"{sub_id}_eyes_open_control.pkl"
    if return_EC_resting:
        raw_ec = load_raw(processed_data_path, sub_id, condition="eyes_closed")
        EC_save_fname = f"{sub_id}_eyes_closed.pkl" if not control_regions else f"{sub_id}_eyes_closed_control.pkl"
    # If processing epochs, check directory for count
    if return_zepochs:
        zepochs_save_fname = f"{sub_id}_epochs.pkl" if not control_regions else f"{sub_id}_epochs_control.pkl"

    # Preallocate
    label_ts_EO, label_ts_EC, label_ts_Epochs = None, None, None
    sub_id_if_nan = None
    # If desired and eyes open resting data not yet processed, process it
    if return_EO_resting and not os.path.exists(
        f"{EO_resting_save_path}/{EO_save_fname}"
    ):
        label_ts_EO, sub_id_if_nan = compute_fwd_and_inv(
            sub_id,
            "EO",
            snr,
            trans,
            src,
            bem,
            raw_eo,
            noise_var,
            labels,
            EO_resting_save_path,
            EO_save_fname,
            method=method,
            average_dipoles=True,
            save_stc_mat=save_stc_mat,
            save_inv=save_inv,
        )

    # If desired and eyes closed resting data not yet processed, process it
    if return_EC_resting and not os.path.exists(
        f"{EC_resting_save_path}/{EC_save_fname}"
    ):
        label_ts_EC, sub_id_if_nan = compute_fwd_and_inv(
            sub_id,
            "EC",
            snr,
            trans,
            src,
            bem,
            raw_ec,
            noise_var,
            labels,
            EC_resting_save_path,
            EC_save_fname,
            method=method,
            average_dipoles=True,
            save_stc_mat=save_stc_mat,
            save_inv=save_inv,
        )

    # If desired and epochs not yet processed, Z-score and source localize
    if return_zepochs:
        if not save_stc_mat and not os.path.exists(
            f"{zscored_epochs_save_path}/{zepochs_save_fname}"
        ):
            print("Z-scoring epochs...")
            zepochs = zscore_epochs(sub_id, processed_data_path, tmin, raw)
            # print shape of zepochs
            print(zepochs.get_data().shape)

            print("Source localizing epochs...")
            label_ts_Epochs, sub_id_if_nan = compute_fwd_and_inv(
                sub_id,
                "epochs",
                snr,
                trans,
                src,
                bem,
                zepochs,
                noise_cov,
                labels,
                zscored_epochs_save_path,
                zepochs_save_fname,
                method=method,
                average_dipoles=True,
                save_stc_mat=save_stc_mat,
                save_inv=save_inv,
            )
        if save_stc_mat:  # for save mat overwrite existing folder
            print(zscored_epochs_save_path, zepochs_save_fname)
            print(os.path.exists(f"{zscored_epochs_save_path}/{zepochs_save_fname}"))

            print("Z-scoring epochs...")
            zepochs = zscore_epochs(sub_id, processed_data_path, tmin, raw)
            # print shape of zepochs
            print(zepochs.get_data().shape)

            print("Source localizing epochs...")
            label_ts_Epochs, sub_id_if_nan = compute_fwd_and_inv(
                sub_id,
                "epochs",
                snr,
                trans,
                src,
                bem,
                zepochs,
                noise_cov,
                labels,
                zscored_epochs_save_path,
                zepochs_save_fname,
                method=method,
                average_dipoles=True,
                save_stc_mat=save_stc_mat,
                save_inv=save_inv,
            )

    return (label_ts_Epochs, label_ts_EO, label_ts_EC), sub_id_if_nan
