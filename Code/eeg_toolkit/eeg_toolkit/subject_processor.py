from glob import glob
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_morlet, AverageTFRArray
from IPython.display import clear_output
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.2)

class Subject:
    def __init__(self, subject_id):
        assert isinstance(subject_id, str), "Subject ID must be a string"
        self.subject_id = subject_id
        self.response = None

    def __str__(self):
        return f"Subject ID: {self.subject_id}, Response: {self.response}"


class SubjectProcessor:
    def __init__(self, paths_dict, roi_acronyms):
        self.yes_list = []
        self.no_list = []
        self.maybe_list = []

        # Define paths and settings
        self.paths_dict = paths_dict
        self.processed_data_path = self.paths_dict["processed_data_path"]
        self.stc_path = self.paths_dict["stc_path"]
        self.EO_resting_data_path = self.paths_dict["EO_resting_data_path"]
        self.zscored_epochs_data_path = self.paths_dict["zscored_epochs_data_path"]

        self.sfreq = 400  # Hz
        self.roi_acronyms = roi_acronyms

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

        # # Eyes Open STC
        # stc_eo_fname = glob(f"{self.EO_resting_data_path}/{this_sub_id}_eyes_open.pkl")
        # # [0]
        # stc_eo = pickle.load(open(stc_eo_fname[0], "rb"))
        # stc_eo = np.array(stc_eo)

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

        # Plot evoked TFR
        fig, axes = plt.subplots(6 , 2, figsize=(12, 6))
        for i, roi in enumerate(self.roi_acronyms):
            ax = axes[i//2, i%2]
            tfr.plot(
                baseline=(-0.2, 0.0),
                tmin=0.0,
                tmax=1.0,
                picks=[i],
                mode="zscore",
                title="Representative Evoked Stimulus Response (Chronic Pain subject)",
                yscale="linear",
                colorbar=False,
                axes=ax,
            )
            ax.set_title(roi)
            
            # Don't set ylabels
            if i > 0:
                ax[i].set_ylabel("")

        # Set labels
        ax[6].set_xlabel("Time (s)")
        ax
        ax.set_xlabel(" ")        
        fig.ylabel()
        fig.colorbar(ax.get_images()[0])
        # fig.tight_layout()
        plt.show()

        # Plot averaged raw trace from evoked
        # Calculate the average across trials (axis=0)
        average_trace = np.mean(stc_epo_array, axis=0)

        # Plot the average trace for each channel
        fig, ax = plt.subplots(6, 2, figsize=(12, 6))

        # Calculate the time vector
        time_range = (-2.5, 2.5)  # Time range in seconds
        timepoints = np.linspace(time_range[0], time_range[1], average_trace.shape[1])

        # Find the index for t=0
        zero_index = np.where(timepoints == 0)[0][0]

        # Plot the average trace
        # Choose the channel you want to plot
        channel = 0
        plt.plot(
            timepoints, 
            average_trace[channel], 
            label=f'Channel {epochs.info["ch_names"][channel]}'
        )

        # Add a vertical red line at t=0
        plt.axvline(x=zero_index, color="red", linestyle="--", label="t=0")

        # Add labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Average trace of Hand 256 mN Trials")
        plt.legend()
        plt.show()

    def get_user_response(self):
        response = input(
            "Is this subject a candidate for representative TFR? (yes/no/maybe): "
        )
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

        # Clear output for next subject
        clear_output(wait=True)

    def display_results(self):
        print("Yes List:", self.yes_list)
        print("No List:", self.no_list)
        print("Maybe List:", self.maybe_list)


def main():
    processor = SubjectProcessor()

    # Example usage: iterate through the subjects
    for sub_id in ["018"]:
        subject = Subject(sub_id)
        processor.plot_subject(subject)
        processor.process_response(subject)

    processor.display_results()


if __name__ == "__main__":
    main()
