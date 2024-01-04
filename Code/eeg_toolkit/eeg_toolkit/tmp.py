
def save_label_time_course(
    sub_id,
    condition,
    snr,
    trans,
    src,
    bem,
    raw,
    noise_cov,
    labels,
    save_path,
    average_dipoles=True,
):

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

    # Make the inverse operator and apply it to the sensor space data
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        raw.info, fwd, noise_cov, verbose=True
    )

    stc = mne.minimum_norm.apply_inverse_raw(
        raw, inverse_operator, method="dSPM", **apply_inverse_rawEC_kwargs
    )

    # Save to file
    src = inverse_operator["src"]
    inverse_operator_zepo = mne.minimum_norm.make_inverse_operator(
        zepochs.info, fwd, noise_cov, verbose=True
    )
    clear_display()

    # Apply inverse to epochs
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