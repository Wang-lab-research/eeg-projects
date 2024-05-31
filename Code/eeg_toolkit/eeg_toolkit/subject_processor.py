from pathlib import Path
from glob import glob
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_morlet, AverageTFRArray


class Subject:
    def __init__(self, subject_id):
        assert isinstance(subject_id, str), "Subject ID must be a string"
        self.subject_id = subject_id
        self.response = None

    def __str__(self):
        return f"Subject ID: {self.subject_id}, Response: {self.response}"


class SubjectProcessor:
    def __init__(self):
        self.yes_list = []
        self.no_list = []
        self.maybe_list = []

        # Define paths and settings
        self.sfreq = 400  # Hz
        self.data_dir = Path("../../Data")
        self.processed_data_path = self.data_dir / "Processed Data"
        self.stc_path = self.data_dir / "Source Time Courses (MNE)"
        self.EO_resting_data_path = self.stc_path / "Eyes Open"
        self.zscored_epochs_data_path = self.stc_path / "zscored_Epochs"
        self.sub_ids_CP = [
            "018",
            "022",
            "024",
            "031",
            "032",
            "034",
            "036",
            "039",
            "040",
            "045",
            "046",
            "052",
            "020",
            "021",
            "023",
            "029",
            "037",
            "041",
            "042",
            "044",
            "048",
            "049",
            "050",
            "056",
        ]
        self.roi_acronyms = [
            "rACC-lh",
            "dACC-lh",
            "S1-lh",
            "insula-lh",
            "dlPFC-lh",
            "mOFC-lh",
            "rACC-rh",
            "dACC-rh",
            "S1-rh",
            "insula-rh",
            "dlPFC-rh",
            "mOFC-rh",
        ]

    def receive_input(self):
        subject_id = input("Enter subject ID: ")
        return Subject(subject_id)

    def plot_subject(self, subject):
        assert isinstance(subject, Subject), "Input must be an instance of Subject"
        this_sub_id = subject.subject_id

        # Load data
        epo_fname = glob(f"{self.processed_data_path}/{this_sub_id}*epo.fif")[0]
        epochs = mne.read_epochs(epo_fname)

        stc_epo_fname = glob(
            f"{self.zscored_epochs_data_path}/{this_sub_id}_epochs.pkl"
        )[0]
        stc_epo = pickle.load(open(stc_epo_fname, "rb"))
        stc_epo = np.array(stc_epo)

        stim_fname = glob(f"{self.processed_data_path}/{this_sub_id}*stim_labels.mat")[
            0
        ]
        stim_labels = sio.loadmat(stim_fname)["stim_labels"][0]

        # Select just hand 256 mN condition (label=3)
        stc_epo_array = stc_epo[stim_labels == 3]

        # Define parameters for the TFR computation
        freqs = np.logspace(*np.log10([1, 100]), num=50)
        n_cycles = freqs / 2.0

        # Construct Epochs info
        info = mne.create_info(
            ch_names=self.roi_acronyms, sfreq=self.sfreq, ch_types="eeg"
        )

        # Compute TFR
        power = tfr_array_morlet(
            stc_epo_array,
            sfreq=self.sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="avg_power",
            zero_mean=False,
        )

        tfr = AverageTFRArray(
            info=info,
            data=power,
            times=epochs.times,
            freqs=freqs,
            nave=stc_epo_array.shape[0],
        )

        # Plot TFR
        tfr.plot(
            baseline=(-2.5, 0.0),
            tmin=0.0,
            tmax=1.0,
            picks=[0],
            mode="zscore",
            title=f"Representative TFR of Subject {this_sub_id}",
        )
        plt.show()

    def get_user_response(self):
        response = input("Is this subject interesting? (yes/no/maybe): ")
        assert response.lower() in [
            "yes",
            "no",
            "maybe",
        ], "Response must be 'yes', 'no', or 'maybe'"
        return response.lower()

    def process_response(self, subject):
        response = self.get_user_response()
        if response == "yes":
            self.yes_list.append(subject.subject_id)
            subject.response = "yes"
        elif response == "no":
            self.no_list.append(subject.subject_id)
            subject.response = "no"
        elif response == "maybe":
            self.maybe_list.append(subject.subject_id)
            subject.response = "maybe"
        else:
            print("Invalid response. Please enter yes, no, or maybe.")
            self.process_response(subject)

    def display_results(self):
        print("Yes List:", self.yes_list)
        print("No List:", self.no_list)
        print("Maybe List:", self.maybe_list)


def main():
    processor = SubjectProcessor()

    # Example usage: iterate through the subjects
    for sub_id in processor.sub_ids_CP:
        subject = Subject(sub_id)
        processor.plot_subject(subject)
        processor.process_response(subject)

    processor.display_results()


if __name__ == "__main__":
    main()
