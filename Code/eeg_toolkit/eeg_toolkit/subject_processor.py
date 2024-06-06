from glob import glob
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_morlet, AverageTFRArray
import seaborn as sns
from typing import Dict, List, Union
from IPython.display import clear_output

sns.set(style="white", font_scale=1.5)


class Subject:
    def __init__(self, subject_id: str):
        assert isinstance(subject_id, str), "Subject ID must be a string"
        self.subject_id = subject_id
        self.response = None

        sub_ids = {
            "CP": [
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
            ],
            "HC": [
                "C10",
                "C11",
                "C12",
                "C13",
                "C14",
                "C15",
                "C16",
                "C17",
                "C18",
                "C19",
                "C2.",
                "C24",
                "C25",
                "C26",
                "C27",
                "C3.",
                "C6.",
                "C7.",
                "C9.",
            ],
            "WSP": [
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
            ],
            "LP": [
                "020",
                "021",
                "023",
                "029",
                "044",
                "037",
                "041",
                "042",
                "048",
                "049",
                "050",
                "056",
            ],
        }

        # Assign group from sub_ids dict keys if sub_id is in any of the keys
        self.group = next((key for key in sub_ids if subject_id in sub_ids[key]), None)

    def __str__(self):
        return f"Subject ID: {self.subject_id}, Group: {self.group}, Response: {self.response}"


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

    def _fill_nan_channels(self, epochs):
        incomplete_ch_names = epochs.info["ch_names"]
        complete_ch_names = [
            "Fp1",
            "Fpz",
            "Fp2",
            "AF3",
            "AF4",
            "F11",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "F12",
            "FT11",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "FT12",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "M1",
            "M2",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
            "Cb1",
            "Cb2",
        ]
        complete_ch_names = [ch_name.upper() for ch_name in complete_ch_names]
        missing_ch_ids = [
            i
            for i in range(len(complete_ch_names))
            if complete_ch_names[i] not in incomplete_ch_names
        ]

        data = epochs.get_data(copy=False)
        data = np.insert(data, missing_ch_ids, np.nan, axis=1)

        info = mne.create_info(
            ch_names=complete_ch_names, sfreq=self.sfreq, ch_types="eeg"
        )

        epochs = mne.EpochsArray(data, info)
        return epochs

    def _load_epochs(self, subject_id: str):
        print(f"\nLoading Epochs for {subject_id}...")
        epo_fname = glob(f"{self.processed_data_path}/{subject_id}*epo.fif")[0]
        epochs = mne.read_epochs(epo_fname)
        print(f"Loaded {len(epochs)} epochs")
        assert isinstance(
            epochs, mne.epochs.EpochsFIF
        ), "Input must be an Epochs object"

        if len(epochs.info["ch_names"]) < 64:
            epochs = self._fill_nan_channels(epochs)
        evoked = np.nanmean(epochs.get_data(copy=False), axis=0)
        sem = np.nanstd(epochs.get_data(copy=False), axis=0) / np.sqrt(len(epochs))
        return epochs, evoked, sem

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

        stc_epo_array = np.nanmean(
            stc_epo[stim_labels == 3], axis=0
        )  # average over hand trials

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

        evoked_data_arrays = []
        sem_epochs_per_sub = []
        stc_epo_arrays = []
        stc_resting_arrays = []
        for subject in subjects_list:
            this_sub_id = subject.subject_id
            epochs, evoked, sem = self._load_epochs(this_sub_id)
            evoked_data_arrays.append(evoked)
            sem_epochs_per_sub.append(sem)

            stc_epo_array = self._load_stc_epochs(this_sub_id)
            stc_epo_arrays.append(stc_epo_array)

            stc_resting = None
            stc_resting_arrays.append(stc_resting)

        # combine data across subjects
        stc_epo_array = np.nanmean(np.array(stc_epo_arrays), axis=0)
        if stc_epo_array.ndim != 3:
            stc_epo_array = np.expand_dims(stc_epo_array, axis=0)
        stc_resting = (
            np.nanmean(np.array(stc_resting_arrays), axis=0)
            if stc_resting is not None
            else None
        )
        evoked_data_arrays = np.array(evoked_data_arrays)
        sem_epochs_per_sub = np.array(sem_epochs_per_sub)

        return (
            epochs,
            evoked_data_arrays,
            sem_epochs_per_sub,
            stc_epo_array,
            stc_resting,
        )

    def _compute_tfr(self, subjects: Union[Subject, SubjectGroup]) -> AverageTFRArray:
        epochs, _, _, stc_epo_array, stc_resting = self._load_complete_data(subjects)

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

    def _plot_tfr(
        self,
        tfr: AverageTFRArray,
        baseline: tuple,
        title: str,
        time_range: tuple = (-0.2, 0.8),
        vlim: tuple = (None, None),
    ):
        fig, axes = plt.subplots(6, 2, figsize=(12, 16), sharex=False, sharey=True)
        for i, roi in enumerate(self.roi_acronyms):
            col = i // 6
            row = i % 6
            ax = axes[row, col]
            tfr.plot(
                baseline=baseline,
                tmin=time_range[0],
                tmax=time_range[1],
                picks=roi,
                mode="zscore",
                title=title,
                yscale="linear",
                colorbar=True,
                vlim=vlim,
                cmap="turbo",
                axes=ax,
                show=False,
            )
            ax.set_title(roi)
            if i > 0:
                ax.set_ylabel("")

            # add stimulus onset line
            ax.axvline(x=0, color="red", linestyle="--", label="Stimulus Onset")

        clear_output(wait=True)
        fig.tight_layout()
        plt.show()

    def _plot_trace(
        self,
        subjects: Union[Subject, SubjectGroup],
        epochs,
        evoked_data_arrays,
        sem_epochs_per_sub,
        channel: str,
        time_range: tuple,
    ):
        evoked_averaged = np.nanmean(evoked_data_arrays, axis=0)
        sem_averaged = np.nanmean(sem_epochs_per_sub, axis=0)

        sfreq = self.sfreq
        total_duration = evoked_averaged.shape[1] / sfreq
        time_min = -total_duration / 2

        sample_start = int((time_range[0] - time_min) * sfreq)
        sample_end = int((time_range[1] - time_min) * sfreq)

        sample_start = max(sample_start, 0)
        sample_end = min(sample_end, evoked_averaged.shape[1])

        timepoints = np.linspace(
            time_range[0], time_range[1], sample_end - sample_start
        )

        # Find the index of the channel
        channel_index = (
            epochs.info["ch_names"].index(channel)
            if channel in epochs.info["ch_names"]
            else epochs.info["ch_names"].index(f"{channel.upper()}")
        )

        # Get data within the time range
        evoked_averaged = evoked_averaged[channel_index, sample_start:sample_end] * 1e7
        sem_averaged = sem_averaged[channel_index, sample_start:sample_end] * 1e7

        plt.figure(figsize=(10, 5))
        plt.plot(
            timepoints,
            evoked_averaged,
            label=f"Channel {channel}",
        )

        # Plot SEM as shaded area
        plt.fill_between(
            timepoints,
            evoked_averaged - sem_averaged,
            evoked_averaged + sem_averaged,
            color="b",
            alpha=0.2,
            label="SEM",
        )

        plt.axvline(x=0, color="red", linestyle="--", label="Stimulus Onset")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.title(
            f"Grand Average of Evoked Response ({subjects.group})"
            if isinstance(subjects, Subject)
            else f"Grand Average of Group-Averaged Evoked Response ({subjects.subjects[0].group})"
        )
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
        vlim=None,
    ):
        tfr = self._compute_tfr(subjects)

        epochs, evoked_data_arrays, sem_epochs_per_sub, stc_epo_array, stc_resting = (
            self._load_complete_data(subjects)
        )

        title = (
            f"Time-Frequency Representation of Evoked  Response ({subjects.group})"
            if isinstance(subjects, Subject)
            else f"Time-Frequency Representation of Group-Averaged Evoked Chronic Pain Response ({subjects.subjects[0].group})"
        )

        self._plot_tfr(tfr, baseline, title, time_range, vlim)
        self._plot_trace(
            subjects, epochs, evoked_data_arrays, sem_epochs_per_sub, channel, time_range
        )

        return tfr, evoked_data_arrays

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
