import numpy as np
import mne


def bp_gen(
    label_ts,
    sfreq,
    fmin,
    fmax,
    tmin=None,
    tmax=None,
):
    """
    Generate band-pass filtered data using MNE library.

    :param label_ts: list of time series data
    :param sfreq: sampling frequency
    :param fmin: minimum frequency of the passband
    :param fmax: maximum frequency of the passband
    :return: generator yielding band-pass filtered data
    """
    filtered_label_ts = []
    for ts in label_ts:
        # crop the data between tmin and tmax
        if tmin is not None or tmax is not None:
            ts = ts[..., int(np.round(tmin * sfreq)) : int(np.round(tmax * sfreq))]

        print(f"ts shape = {ts.shape}")

        filtered_label_ts.append(
            mne.filter.filter_data(
                ts, sfreq, fmin, fmax, phase="zero-double", method="iir"
            )
        )
    return np.asarray(filtered_label_ts)
