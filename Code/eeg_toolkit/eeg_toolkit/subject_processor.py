from glob import glob
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_morlet, AverageTFRArray
from IPython.display import clear_output
import seaborn as sns
from typing import Dict, List, Union

sns.set(style="white", font_scale=1.5)


class Subject:
    def __init__(self, subject_id: str):
        assert isinstance(subject_id, str), "Subject ID must be a string"
        self.subject_id = subject_id
        self.response = None

    def __str__(self):
        return f"Subject ID: {self.subject_id}, Response: {self.response}"


class SubjectGroup:
    def __init__(self, subjects: List[Subject]):
        assert isinstance(subjects, list), "Input must be a list"
        assert all(
            [isinstance(el, Subject) for el in subjects]
        ), "Input must be a list of Subjects"
        self.subjects = subjects

    def __str__(self):
        return f"Subjects: {self.subjects}"


class SubjectProcessor:
    def __init__(self, paths_dict: Dict[str, str], roi_acronyms: List[str]):
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

    def _load_epochs(self, subject_id: str):
        print(f"\nLoading Epochs for {subject_id}...")
        epo_fname = glob(f"{self.processed_data_path}/{subject_id}*epo.fif")[0]
        epochs = mne.read_epochs(epo_fname)
        print(f"Loaded {len(epochs)} epochs")
        assert isinstance(
            epochs, mne.epochs.EpochsFIF
        ), "Input must be an Epochs object"
        average_trace = np.mean(epochs.get_data(copy=False), axis=0)
        return epochs, average_trace

    def _load_stc_epochs(self, subject_id: str):
        print(f"Loading STC epochs for {subject_id}...")
        stc_epo_fname = glob(
            f"{self.zscored_epochs_data_path}/{subject_id}_epochs.pkl"
        )[0]
        stc_epo = pickle.load(open(stc_epo_fname, "rb"))
        stc_epo = np.array(stc_epo)

        stim_fname = glob(f"{self.processed_data_path}/{subject_id}*stim_labels.mat")[0]
        stim_labels = sio.loadmat(stim_fname)["stim_labels"][0]

        print(f"Loaded {len(stim_labels)} evoked trials")
        
        # Select just hand 256 mN condition (label=3)
        stc_epo_array = stc_epo[stim_labels == 3]
        assert isinstance(stc_epo_array, np.ndarray), "Input must be an array"
        return stc_epo_array

    def _load_complete_data(
        self,
        subjects: Union[Subject, SubjectGroup],
    ):
        assert isinstance(subjects, Subject) or isinstance(
            subjects, SubjectGroup
        ), "Input must be an instance of Subject or SubjectGroup"

        if isinstance(subjects, SubjectGroup):
            subjects_list = [subject for subject in subjects.subjects]
        elif isinstance(subjects, Subject):
            subjects_list = [subjects]

        average_epochs_arrays = []
        stc_epo_arrays = []
        stc_eos = []
        for subject in subjects_list:
            this_sub_id = subject.subject_id
            epochs, average_trace = self._load_epochs(this_sub_id)
            average_epochs_arrays.append(average_trace)

            stc_epo_array = self._load_stc_epochs(this_sub_id)
            stc_epo_arrays.append(stc_epo_array)

            stc_eo = None
            stc_eos.append(stc_eo)

        average_epoch_data = np.mean(np.array(average_epochs_arrays), axis=0)
        stc_epo_array = np.mean(np.array(stc_epo_arrays), axis=0)
        stc_eo = np.mean(np.array(stc_eos), axis=0) if stc_eo is not None else None

        return epochs, average_epoch_data, stc_epo_array, stc_eo

    def _compute_tfr(self, subjects: Union[Subject, SubjectGroup]) -> AverageTFRArray:
        assert isinstance(subjects, Subject) or isinstance(
            subjects, SubjectGroup
        ), "Input must be an instance of Subject or SubjectGroup"
        epochs, _, stc_epo_array, stc_eo = self._load_complete_data(subjects)

        freqs = np.logspace(*np.log10([1, 100]), num=50)
        n_cycles = freqs / 2.0

        info = mne.create_info(
            ch_names=self.roi_acronyms, sfreq=self.sfreq, ch_types="eeg"
        )

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

        return tfr

    def _plot_tfr(self, tfr: AverageTFRArray, baseline: tuple, title: str):
        fig, axes = plt.subplots(6, 2, figsize=(12, 16), sharex=False, sharey=True)
        for i, roi in enumerate(self.roi_acronyms):
            col = i // 6
            row = i % 6
            ax = axes[row, col]
            tfr.plot(
                baseline=baseline,
                tmin=0.0,
                tmax=1.0,
                picks=roi,
                mode="zscore",
                title=title,
                yscale="linear",
                colorbar=True,
                vlim=(-5, 5),
                cmap="turbo",
                axes=ax,
                show=False,
            )
            ax.set_title(roi)
            if i > 0:
                ax.set_ylabel("")

        clear_output(wait=True)
        fig.tight_layout()
        plt.show()

    def _plot_trace(self, epochs, average_epoch_data, channel: str, time_range: tuple):
        if isinstance(average_epoch_data, np.ndarray):
            average_trace = average_epoch_data
        else:
            average_trace = np.mean(epochs.get_data(copy=False), axis=0)

        sfreq = self.sfreq
        total_duration = average_trace.shape[1] / sfreq
        time_min = -total_duration / 2

        sample_start = int((time_range[0] - time_min) * sfreq)
        sample_end = int((time_range[1] - time_min) * sfreq)

        sample_start = max(sample_start, 0)
        sample_end = min(sample_end, average_trace.shape[1])

        timepoints = np.linspace(
            time_range[0], time_range[1], sample_end - sample_start
        )

        channel_index = epochs.info["ch_names"].index(channel)
        average_trace = average_trace[channel_index, sample_start:sample_end]*1e6

        plt.figure(figsize=(8, 4))
        plt.plot(
            timepoints,
            average_trace,
            label=f"Channel {channel}",
        )

        plt.axvline(x=0, color="red", linestyle="--", label="Stimulus Onset")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.title("Grand Average of Evoked Chronic Pain Response")
        plt.tight_layout()
        plt.legend()
        plt.xlim(time_range)
        plt.show()

    def plot_TFR_and_trace(
        self,
        subjects: Union[Subject, SubjectGroup],
        channel="Fp1",
        baseline=(-2.5, 0.0),
        time_range=(-0.2, 0.8),
    ):
        assert isinstance(subjects, Subject) or isinstance(
            subjects, SubjectGroup
        ), "Input must be an instance of Subject or SubjectGroup"

        tfr = self._compute_tfr(subjects)
        epochs, average_epoch_data, _, stc_eo = self._load_complete_data(subjects)

        title = (
            "Time-Frequency Representation of Evoked Chronic Pain Response"
            if isinstance(subjects, Subject)
            else "Time-Frequency Representation of Group-Averaged Evoked Chronic Pain Response"
        )

        self._plot_tfr(tfr, baseline, title)
        self._plot_trace(epochs, average_epoch_data, channel, time_range)

    def _get_user_response(self) -> str:
        response = input(
            "Is this subject a candidate for representative TFR? (yes/no/maybe): "
        )
        assert response.lower() in [
            "yes",
            "no",
            "maybe",
        ], "Response must be 'yes', 'no', or 'maybe'"
        return response.lower()

    def process_response(self, subject: Subject):
        response = self._get_user_response()
        if response == "yes":
            self.yes_list.append(subject.subject_id)
            subject.response = "yes"
        elif response == "no":
            self.no_list.append(subject.subject_id)
            subject.response = "no"
        elif response == "maybe":
            self.maybe_list.append(subject.subject_id)
            subject.response = "maybe"

        clear_output(wait=True)

    def display_results(self):
        print("Yes List:", self.yes_list)
        print("No List:", self.no_list)
        print("Maybe List:", self.maybe_list)


def main():
    paths_dict = {
        "processed_data_path": "/path/to/processed_data",
        "stc_path": "/path/to/stc",
        "EO_resting_data_path": "/path/to/EO_resting_data",
        "zscored_epochs_data_path": "/path/to/zscored_epochs_data",
    }
    roi_acronyms = [
        "ROI1",
        "ROI2",
        "ROI3",
        "ROI4",
        "ROI5",
        "ROI6",
        "ROI7",
        "ROI8",
        "ROI9",
        "ROI10",
        "ROI11",
        "ROI12",
    ]

    processor = SubjectProcessor(paths_dict, roi_acronyms)

    for sub_id in ["018"]:
        subject = Subject(sub_id)
        processor.plot_TFR_and_trace(subject, channel="Fp1", time_range=(-0.2, 0.8))
        processor.process_response(subject)

    processor.display_results()


if __name__ == "__main__":
    main()
