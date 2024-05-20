import os
import mne
import h5py
from glob import glob
import pandas as pd
import numpy as np
from IPython import display
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels

# Global variables
RANDOM_STATE = 42


def clear_display():
    display.clear_output(wait=True)


# Function for loading hdf5 file from parent data folder given sub id
def load_file(sub_id, data_path, extension="hdf5"):
    """
    Load a file based on the subject ID, data path, and file extension.

    Parameters:
    - sub_id: The subject ID to load the file for.
    - data_path: The path to the data folder.
    - extension: The file extension to load (default is "hdf5").

    Returns:
    - The loaded file based on the specified extension.
    """
    for folder in os.listdir(data_path):
        if sub_id in folder:
            sub_id = folder
            break
    if extension == "hdf5":
        h5_files = glob(os.path.join(data_path, folder, f"*.{extension}"))
        return h5py.File(h5_files[0], "r")
    elif extension == "edf":
        edf_files = glob(os.path.join(data_path, folder, "*.edf")) + glob(
            os.path.join(data_path, folder, "*.EDF")
        )
        return mne.io.read_raw_edf(edf_files[0], preload=True)
    elif extension == "csv":
        csv_files = glob(os.path.join(data_path, folder, "*.csv"))
        return pd.read_csv(csv_files[0])


def remove_trailing_zeros(raw, sub_id, sfreq):
    """
    Removes trailing zeros from raw data channels.

    Parameters:
    - raw: The raw data object containing time-series data.
    - sub_id: Subject identifier.
    - sfreq: Sampling frequency.

    Returns:
    - raw: The potentially modified raw data object after cropping.
    - need_crop: A boolean indicating if cropping was performed.
    """
    raw_dur = raw.times[-1]
    raw_data = raw.get_data()
    need_crop = False

    print(f"Looking for trailing zeros in subject {sub_id}")

    zero_count = 0
    ch = raw_data[0]
    for i in range(len(ch)):
        if ch[i] == 0.0:
            zero_count += 1
            if zero_count >= 100:
                start_index = i - (zero_count - 1)
                end_index = len(ch)
                print(
                    f"{zero_count} consecutive zeros found starting at index {start_index}"
                )
                zeros_dur = (end_index - start_index) / sfreq
                print(f"Duration: {zeros_dur} sec")
                need_crop = True
                break
        else:
            zero_count = 0
    if need_crop:
        print("Need to crop trailing zeros")
        raw = raw.crop(tmin=0, tmax=raw_dur - np.ceil(zeros_dur), include_tmax=False)

    return raw, need_crop


def to_raw(data_path, sub_id, save_path, eog_flag):
    """
    Preprocess raw EDF data to filtered FIF format.
    """
    # Load hdf5
    h5_file = load_file(sub_id, data_path, extension="hdf5")
    edf_file = load_file(sub_id, data_path, extension="edf")

    # Get data
    data = h5_file["RawData"]["Samples"]

    # Create new mne.io.Raw object
    raw = mne.io.RawArray(data[:].T, edf_file.info)

    # Crop the first sample
    sfreq = raw.info["sfreq"]

    # read data, set EOG channel, and drop unused channels
    print(f"Subject {sub_id}\nreading raw file...")

    # Assuming `raw`, `sub_id`, and `raw_sfreq` are already defined:
    raw_cropped, was_cropped = remove_trailing_zeros(raw, sub_id, sfreq)
    if was_cropped:
        print("Data was cropped to remove trailing zeros.")
        raw = raw_cropped

    # rename channel 65 to EOG
    if "65" in raw.info["ch_names"]:
        raw.rename_channels({"65": "EOG"})
        raw.set_channel_types({"EOG": "eog"})

    # drop reference channels A1 and A2
    if "A1" in raw.info["ch_names"]:
        raw.drop_channels(["A1", "A2"])

    # if channel names are numeric, drop them
    raw.drop_channels([ch for ch in raw.ch_names if ch.isnumeric()])

    ## Read data, set EOG channel, and drop unused channels

    # For 64 channel gTec cap
    mont1020 = mne.channels.make_standard_montage("standard_1020")

    # Rename capitalized channels to lowercase
    print("Renaming capitalized channels to lowercase...")
    for i, ch in enumerate(raw.info["ch_names"]):
        if "FP" in ch:
            raw.rename_channels({ch: "Fp" + ch[2:]})

    # Choose what channels you want to keep
    # Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
    kept_channels = raw.info["ch_names"][:64]
    ind = [
        i
        for (i, channel) in enumerate(mont1020.ch_names)
        if channel.lower() in map(str.lower, kept_channels)
    ]
    mont1020_new = mont1020.copy()
    # Keep only the desired channels
    mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x + 3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    mont1020_new.dig = mont1020.dig[0:3] + kept_channel_info
    raw = raw.set_montage(mont1020_new)

    # high level inspection
    print(raw.ch_names)
    print(len(raw.ch_names))

    # apply notch filter
    print(f"Subject {sub_id}\napplying notch filter...")
    raw = raw.notch_filter(60.0, notch_widths=3)
    clear_display()

    # apply bandpass filter
    print(f"Subject {sub_id}\napplying bandpass filter...")
    raw = raw.filter(l_freq=1.0, h_freq=100.0)
    clear_display()

    # find bad channels automatically
    print(f"Subject {sub_id}\nremoving bad channels...")
    raw_pyprep = NoisyChannels(raw, random_state=RANDOM_STATE)
    raw_pyprep.find_all_bads(ransac=False)
    raw.info["bads"] = raw_pyprep.get_bads()
    print(f"Subject {sub_id} bad channels: {raw.info['bads']}")
    # raw.interpolate_bads()
    clear_display()

    # fit ICA
    print(f"Subject {sub_id}\nfitting ICA...")
    num_goods = len(raw.ch_names) - len(raw.info["bads"]) - 1  # adjust for EOG
    ica = ICA(
        n_components=int(np.floor(num_goods / 2)),
        random_state=RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw)
    clear_display()

    # find EOG artifacts
    print(raw.ch_names)
    if eog_flag:
        print(f"Subject {sub_id}\nfinding EOG artifacts...")
        eog_indices, eog_scores = ica.find_bads_eog(raw, threshold="auto")
        ica.exclude = eog_indices
    else:
        ica.exclude = [0, 1]
        raw.drop_channels("EOG")
    clear_display()

    # apply ICA
    print(f"Subject {sub_id}\napplying ICA...")
    ica.apply(raw)
    clear_display()

    # save copy of data
    save_fname_fif = f"{sub_id}_preprocessed-raw.fif"
    print(f"Saving processed data as '{save_fname_fif}'...")

    # re-reference channels
    print(f"Subject {sub_id}\nre-referencing channels to average...")
    raw, _ = mne.set_eeg_reference(raw, ref_channels="average", copy=True)
    clear_display()

    # crop first second due to high amplitude noise
    raw.crop(tmin=1.0)  # crop first second

    # save raw data
    raw.save(save_path / save_fname_fif, verbose=True, overwrite=True)

    # high level inspection
    print(raw.ch_names)
    print("\nNumber of good channels: ", len(raw.ch_names) - len(raw.info["bads"]))
    print("\nDropped channels: ", raw.info["bads"])

    print("Raw data preprocessing complete.")

    # clear_display()

    return raw


def crop_resting_EO(
    raw, sub_id, data_path, processed_data_path, events=None, event_ids=None
):
    # Get converted EDF file for events
    raw_edf = load_file(sub_id, data_path, extension="edf")
    sfreq = raw.info["sfreq"]

    if events is None and event_ids is None:
        events, event_ids = mne.events_from_annotations(raw_edf)

    # For now get just the events we are interested in, eyes open and stop

    # Set events indicating start and end of resting eyes open
    eyes_open_id = event_ids["KB-Marker-0 (Eyes open) "]
    stop_id = event_ids["KB-Marker-9 (END/STOP test) "]

    # Check for eyes open marker and stop marker either right after or two after the event, to accout for mistakes
    max_time = 320 * sfreq  # maximum time between eyes open and stop
    for i in range(len(events)):
        # local events
        if events[i][2] == eyes_open_id:
            this_event = events[i]
            next_event = events[i + 1]
            following_event = events[i + 2] if len(events) > i + 2 else None

            # get event ids
            this_event_id = this_event[2]
            next_event_id = next_event[2]
            following_event_id = (
                following_event[2] if following_event is not None else None
            )

            # get event times
            this_event_samples = this_event[0]
            next_event_samples = next_event[0]
            following_event_samples = (
                following_event[0] if following_event is not None else None
            )

            # save eyes open times if valid
            if (
                this_event_id == eyes_open_id
                and next_event_id == stop_id
                and (next_event_samples - this_event_samples) < max_time
            ):
                print("\nEyes open found and next event STOP")
                eyes_open_events = [i, i + 1]
                eyes_open_times = [el[0] for el in events[eyes_open_events]]
            elif (
                this_event_id == eyes_open_id
                and following_event_id == stop_id
                and (following_event_samples - this_event_samples) < max_time
            ):
                print("\nEyes open found and FOLLOWING event STOP")
                eyes_open_events = [i, i + 2]
                eyes_open_times = [el[0] for el in events[eyes_open_events]]
            elif this_event_id == eyes_open_id and (
                next_event_id != stop_id or following_event_id != stop_id
            ):
                print("\nEyes open found but NO STOP found")
                eyes_open_events = [
                    i,
                ]
                eyes_open_times = [el[0] for el in events[eyes_open_events]]
                eyes_open_times.append(eyes_open_times[0] + 300 * sfreq)
            else:
                raise ValueError("\nError, an eyes open window cannot be created")

            # Get eyes open times
            eyes_open_times_seconds = [el / sfreq for el in eyes_open_times]
            break

    print(f"\nEyes open times: {eyes_open_times_seconds}")

    # save cropped data
    raw.crop(tmin=eyes_open_times_seconds[0], tmax=eyes_open_times_seconds[-1])

    raw.save(processed_data_path / f"{sub_id}_eyes_open-raw.fif", overwrite=True)

    # crop data
    return raw


def snip_span(raw, t1, t2):
    """
    Extracts the data from raw as numpy array, snip a middle section out based on two time values,
    reconcatenate the numpy array, then create an mne.RawArray object using the raw.info

    Args:
    raw (mne.io.Raw): Input raw data
    t1 (float): Start time of the section to be removed in seconds
    t2 (float): End time of the section to be removed in seconds

    Returns:
    mne.io.RawArray: Processed raw data
    """

    # Extract the data and times from raw
    data, times = raw[:]

    # Convert times to indices
    idx1 = np.argmin(np.abs(times - t1))
    idx2 = np.argmin(np.abs(times - t2))

    # Snip out the middle section and reconcatenate
    processed_data = np.concatenate((data[:, :idx1], data[:, idx2:]), axis=1)

    # Create a new MNE RawArray object with the processed data
    processed_raw = mne.io.RawArray(processed_data, raw.info)

    return processed_raw
