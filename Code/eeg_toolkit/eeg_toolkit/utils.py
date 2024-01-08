import os
import mne
from IPython import display
import glob
import pickle


def clear_display():
    display.clear_output(wait=True)


def load_raw_data(eeg_data_raw_file, eog):
    """
    Load raw EDF data with specified EOG channel.
    """
    return mne.io.read_raw_edf(eeg_data_raw_file, eog=[eog], preload=True)


def set_montage(mne_obj, montage):
    """
    Set custom montage for Raw or Epochs object.
    """
    print("setting custom montage...")
    print(montage)
    if isinstance(montage, str):
        relative_path = os.path.join(os.path.dirname(__file__), montage)
        dig_montage = mne.channels.read_custom_montage(relative_path)
        mne_obj.set_montage(dig_montage, on_missing="ignore")
    else:
        mne_obj.set_montage(montage, on_missing="ignore")


def import_subs(data_path, fname):
    # import sub_ids
    sub_ids = []
    with open(os.path.join(data_path, fname), "r") as file:
        for line in file:
            # Check if the line is not commented out
            if not line.strip().startswith("#"):
                # Extracting the subject ID and ignoring any comments or trailing commas
                sub_id = line.split(",")[0].strip().strip("'")
                if sub_id:
                    sub_ids.append(sub_id)
    return sub_ids


# functions for serialization
def pickle_data(save_path, fname, data):
    with open(os.path.join(save_path, fname), "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {fname} to {save_path}.")


def unpickle_data(data_path):
    fname = glob(os.path.join(data_path, "*.pkl"))
    with open(fname, "rb") as f:
        deserialized_object = pickle.load(f)
    return deserialized_object
