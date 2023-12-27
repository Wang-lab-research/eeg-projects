##################################################################
from utils import *
from mne.datasets import fetch_fsaverage

get_ipython().run_line_magic("matplotlib", "inline")
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
apply_inverse_raw_kwargs = dict(
    lambda2=1.0 / snr**2, verbose=True  # regularizer parameter (λ²)
)
##################################################################


def make_sub_time_win_path(
    sub_id, save_path_cont, save_path_zepo, include_zepochs=True
):
    """
    Make a subject's time window data path.

    Args:
        sub_id (str): The subject ID.
        save_path_cont (str): The path to save continuous data.
        save_path_zepo (str): The path to save zepochs data.
        include_zepochs (bool, optional): Whether to include zepochs data. Defaults to True.

    Returns:
        tuple: A tuple containing the paths for continuous data and zepochs data (if included).
    """
    subpath_cont = os.path.join(save_path_cont, sub_id)
    if not os.path.exists(subpath_cont):  # continuous
        os.mkdir(subpath_cont)

    subpath_zepo = None
    if include_zepochs:
        subpath_zepo = os.path.join(save_path_zepo, sub_id)
        if not os.path.exists(subpath_zepo):  # zepochs
            os.mkdir(subpath_zepo)

    return subpath_cont, subpath_zepo


def to_source(
    sub_id,
    data_path,
    epo_path,
    save_path_cont,
    save_path_zepo,
    roi_names,
    times_tup,
    noise_cov_win,
    include_zepochs=True,
    average_dipoles=True,
):
    """
    Perform source localization on Raw object
    using fsaverage for certain selected labels.
    sub_id: subject ID
    data_path: contains raw and info
    epo_path: specifies epo objects specified with time_win
    roi_names: the roi to extract from STC as a list
    times_tup: contains tmin,tmax,bmax
    noise_cov_win: (rest_min,rest_max). Crop raw during eyes-open resting condition
    include_zepochs: whether to also export z-scored epochs, default True.
    average_dipoles: whether to average source points in each ROI, default True.
    """
    #################################### Read Raw and Epochs & Set montage ###########################################
    zepochs_stc_arr = None
    sub_raw_fname = f"{sub_id}_preprocessed-raw.fif"
    raw_path = os.path.join(data_path, sub_raw_fname)
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    print(sub_raw_fname)
    raw.set_eeg_reference("average", projection=True)

    labels = [
        mne.read_labels_from_annot(subject, regexp=roi, subjects_dir=subjects_dir)[0]
        for roi in roi_names
    ]

    # Extract time window information from tuple arguments
    tmin, tmax, bmax = times_tup
    rest_min, rest_max = noise_cov_win

    # Make subpaths
    subpath_cont, subpath_zepo = make_sub_time_win_path(
        sub_id,
        save_path_cont,
        save_path_zepo,
        include_zepochs=True,
    )

    # Quick check to see if subject already processed
    subpath_cont_count = len(os.listdir(subpath_cont))
    subpath_zepo_count = len(os.listdir(subpath_zepo))
    roi_names_count = len(roi_names)

    if (
        subpath_cont_count < roi_names_count or subpath_zepo_count < roi_names_count
    ):  ##################################### Z-score Epochs then convert to STC ############################################
        if include_zepochs:
            sub_epo_fname = f"{sub_id}_preprocessed-epo.fif"
            epochs_path = os.path.join(epo_path, sub_epo_fname)
            epochs = mne.read_epochs(epochs_path)
            print(sub_epo_fname)

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
            set_montage(zepochs, raw.get_montage())

        ##################################### Crop resting EC and EO ############################################
        # TODO: perform crop for resting state
        # TODO: use different noise covariance for resting state

        ##################################### Compute noise & data covariance ############################################
        raw_crop = raw.copy().crop(tmin=60 * rest_min, tmax=60 * rest_max)
        noise_cov = mne.compute_raw_covariance(raw_crop, verbose=True)

        ################################### Regularize the covariance matrices ##########################################
        noise_cov = mne.cov.regularize(noise_cov, raw_crop.info, eeg=0.1, verbose=True)

        #################################### Compute the forward solution ###############################################
        fwd = mne.make_forward_solution(
            raw.info,
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
            raw.info, fwd, noise_cov, verbose=True
        )

        if include_zepochs:
            inverse_operator_zepo = mne.minimum_norm.make_inverse_operator(
                zepochs.info, fwd, noise_cov, verbose=True
            )
        clear_display()

        ################################# Save source time courses #######################################
        if len(os.listdir(subpath_cont)) < len(roi_names):
            print(sub_raw_fname)
            src_cont = inverse_operator["src"]
            stc_cont = mne.minimum_norm.apply_inverse_raw(
                raw, inverse_operator, method="dSPM", **apply_inverse_raw_kwargs
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
        if include_zepochs:
            if len(os.listdir(subpath_zepo)) < len(roi_names):
                print(sub_epo_fname)
                zepochs_stc = mne.minimum_norm.apply_inverse_epochs(
                    zepochs,
                    inverse_operator_zepo,
                    method="dSPM",
                    **apply_inverse_raw_kwargs,
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
