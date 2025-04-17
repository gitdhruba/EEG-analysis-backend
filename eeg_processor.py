import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
import mne  # type: ignore
import os
from scipy.signal import butter, filtfilt, iirnotch, welch  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from typing import Tuple, List, Optional

bands: List[Tuple[str, float, float]] = [
    ('Delta', 0.5, 4),  # index = 0
    ('Theta', 4, 8),    # index = 1
    ('Alpha', 8, 12),   # index = 2
    ('Beta', 12, 30),   # index = 3
    ('Gamma', 30, 50)   # index = 4
]

# Load the pre-trained model
MODEL_PATH: str = "./binary_classification.pkl"
model = joblib.load(MODEL_PATH)


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    nyquist: float = 0.5 * fs
    low: float = lowcut / nyquist
    high: float = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    if data.ndim == 1:
        filtered_data: np.ndarray = filtfilt(b, a, data)
    else:
        filtered_data: np.ndarray = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=1, arr=data)
    return filtered_data


def remove_artifacts(eeg_data: np.ndarray, fs: int, bandpass: Tuple[float, float] = (1, 50), 
                     notch_freq: float = 50, notch_quality: float = 30) -> np.ndarray:
    def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
        nyquist: float = 0.5 * fs
        low: float = lowcut / nyquist
        high: float = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def apply_notch_filter(data: np.ndarray, freq: float, fs: int, quality: float) -> np.ndarray:
        nyquist: float = 0.5 * fs
        w0: float = freq / nyquist
        b, a = iirnotch(w0, quality)
        return filtfilt(b, a, data)

    lowcut, highcut = bandpass
    filtered_data: np.ndarray = np.apply_along_axis(lambda x: bandpass_filter(x, lowcut, highcut, fs), axis=1, arr=eeg_data)
    cleaned_data: np.ndarray = np.apply_along_axis(lambda x: apply_notch_filter(x, notch_freq, fs, notch_quality), axis=1, arr=filtered_data)
    return cleaned_data


def process_eeg_data(data: np.ndarray, sampling_frequency: int, window_duration: float) -> np.ndarray:
    samples_per_window: int = int(sampling_frequency * window_duration)
    all_windows: List[np.ndarray] = []
    if data.ndim != 2:
        raise ValueError(f"Data must be a 2D array (channels x samples).")

    num_channels, total_samples = data.shape
    num_windows: int = total_samples // samples_per_window
    trimmed_data: np.ndarray = data[:, :num_windows * samples_per_window]
    windows: np.ndarray = trimmed_data.reshape(num_channels, num_windows, samples_per_window).transpose(1, 0, 2)
    all_windows.append(windows)
    combined_windows: np.ndarray = np.concatenate(all_windows, axis=0)
    return combined_windows


def compute_power_features(eeg_data: np.ndarray, fs: int) -> Tuple[np.ndarray, List[List[float]], List[List[float]]]:
    power_features: List[List[List[float]]] = []
    band_frequencies: List[List[float]] = [None] * len(bands)
    f, _ = welch(eeg_data[0][0], fs, nperseg=1024)
    for i in range(len(bands)):
        mask: np.ndarray = (f >= bands[i][1]) & (f <= bands[i][2])
        band_frequencies[i] = f[mask].tolist()

    pxx_avg: List[List[float]] = [[0] * len(b) for b in band_frequencies]

    for window in eeg_data:
        power_window: List[List[float]] = []
        for channel_data in window:
            _, Pxx = welch(channel_data, fs, nperseg=1024)
            power_band: List[float] = []
            for i, (_, f_min, f_max) in enumerate(bands):
                band_mask: np.ndarray = (f >= f_min) & (f <= f_max)
                curr_Pxx: np.ndarray = Pxx[band_mask]
                for j in range(len(band_frequencies[i])):
                    pxx_avg[i][j] += curr_Pxx[j]

                band_power: float = np.trapz(curr_Pxx, band_frequencies[i])
                power_band.append(band_power)
            power_window.append(power_band)
        power_features.append(power_window)

    for pxx_curr in pxx_avg:
        for i in range(len(pxx_curr)):
            pxx_curr[i] /= (eeg_data.shape[0] * eeg_data.shape[1])

    return np.array(power_features), pxx_avg, band_frequencies


def compute_entropy_features(eeg_data: np.ndarray) -> np.ndarray:
    def approximate_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        N: int = len(x)
        phi: List[float] = []
        for m in range(1, m + 1):
            X: np.ndarray = np.array([x[i:i + m] for i in range(N - m + 1)])
            C: np.ndarray = np.array([np.sum(np.abs(X - X[i]) <= r) / (N - m + 1) for i in range(N - m)])
            phi.append(np.mean(np.log(C)))
        return phi[0] - phi[1]

    entropy_features: List[List[float]] = []
    for window in eeg_data:
        entropy_window: List[float] = []
        for channel_data in window:
            entropy_window.append(approximate_entropy(channel_data))
        entropy_features.append(entropy_window)
    return np.array(entropy_features)


def calculate_engagement_index(power_features: np.ndarray) -> np.float64:
    alpha_index: int = 2
    beta_index: int = 3
    theta_index: int = 1

    alpha_power: np.float64 = np.float64(0)
    beta_power: np.float64 = np.float64(0)
    theta_power: np.float64 = np.float64(0)

    for w in power_features:
        for ch in w:
            alpha_power += ch[alpha_index]
            beta_power += ch[beta_index]
            theta_power += ch[theta_index]

    alpha_power /= (power_features.shape[0] * power_features.shape[1])
    beta_power /= (power_features.shape[0] * power_features.shape[1])
    theta_power /= (power_features.shape[0] * power_features.shape[1])

    if (alpha_power + theta_power) == np.float64(0):
        return np.float64(0)
    else:
        return beta_power / (alpha_power + theta_power)


def preprocess_eeg_data(eeg_data: np.ndarray, fs: int) -> Tuple[np.ndarray, List[List[float]], List[List[float]], np.float64]:
    power_features, pxx_avg, band_frequencies = compute_power_features(eeg_data, fs)
    entropy_features: np.ndarray = compute_entropy_features(eeg_data)
    entropy_features = entropy_features[:, :, np.newaxis]
    combined_features: np.ndarray = np.concatenate((power_features, entropy_features), axis=-1)
    n_windows, _, _ = combined_features.shape
    combined_features = combined_features.reshape(n_windows, -1)
    scaler = MinMaxScaler()
    standardized_features: np.ndarray = scaler.fit_transform(combined_features)
    labels: np.ndarray = np.ones((n_windows, 1))
    preprocessed_data: np.ndarray = np.hstack((standardized_features, labels))
    return preprocessed_data, pxx_avg, band_frequencies, calculate_engagement_index(power_features)


def check_load(result: np.ndarray) -> str:
    ones: int = 0
    zeros: int = 0
    for x in result:
        if x == np.float64(1.0):
            ones += 1
        else:
            zeros += 1
    return "Load Present" if ones > zeros else "No Load"


def process_eeg(data: pd.DataFrame, fs : int) -> Tuple[Optional[str], Optional[List[List[float]]], Optional[List[List[float]]], Optional[np.float64]]:
    lowcut: float = 1
    highcut: float = 50
    window_duration: float = 2

    filtered_data: np.ndarray = bandpass_filter(data.to_numpy(), lowcut, highcut, fs)
    f_a_data: np.ndarray = remove_artifacts(filtered_data, fs, bandpass=(1, 50), notch_freq=50, notch_quality=30)
    segmented_data: np.ndarray = process_eeg_data(f_a_data, fs, window_duration)

    features_data, pxx_avg, band_frequencies, ei = preprocess_eeg_data(segmented_data, fs)
    features_data = pd.DataFrame(features_data)
    selected_features: pd.DataFrame = features_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20,
                                                                21, 22, 26, 28, 29, 30, 31, 32, 33, 35, 38, 39, 41, 47, 50, 52,
                                                                54, 55, 57, 58, 59, 60, 64, 70, 71, 79, 81, 82, 83, 89, 94, 95]]
    result: np.ndarray = model.predict(selected_features)
    return check_load(result), pxx_avg, band_frequencies, ei




def get_events_from_vmrk(vmrk_file: str) -> List[Tuple[str, int]]:
    """
    Extracts events from a .vmrk file.

    Args:
        vmrk_file (str): Path to the .vmrk file.

    Returns:
        List[Tuple[str, int]]: A list of tuples containing event descriptions and their corresponding sample indices.
    """
    events: List[Tuple[str, int]] = []

    with open(vmrk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Mk"):  # Check if the line contains marker information
                info: List[str] = line.strip().split(",")  # Split line by comma
                if len(info) >= 4:  # Ensure the line has enough fields
                    desc: str = info[1].strip()  # Extract the event description
                    index: int = int(info[2].strip())  # Extract the sample index
                    events.append((desc, index))  # Append the event to the list

    return events


def process_eeg_file(vhdr_filename: str, vmrk_filename: str, event_desc: str) -> Tuple[Optional[str], Optional[List[dict]], Optional[List[float]], Optional[np.float64]]:
    """
    Processes an EEG file and extracts data for a specific event.

    Args:
        vhdr_filename (str): Path to the .vhdr file.
        vmrk_filename (str): Path to the .vmrk file.
        event_desc (str): Description of the event to extract.

    Returns:
        Tuple[Optional[str], Optional[List[dict]], Optional[List[float]], Optional[np.float64]]:
            - Result of the EEG processing (e.g., "Load Present" or "No Load").
            - Plot data for power spectral density.
            - Cleaned EEG data.
            - Engagement index (EI).
    """
    # Load the raw EEG data from the .vhdr file
    raw = mne.io.read_raw_brainvision(vhdr_filename, preload=True)
    sampling_freq: int = int(os.getenv("FS"))  # sampling frequency
    data, times = raw.get_data(return_times=True)  # Extract data and corresponding time points

    # Extract events from the .vmrk file
    events: List[Tuple[str, float]] = get_events_from_vmrk(vmrk_filename)
    event_times: List[Tuple[str, float]] = [(desc, sample_index / sampling_freq) for desc, sample_index in events]

    # Identify segments corresponding to the specified event
    segments: List[Tuple[float, float]] = [
        (start_time, (event_times[i + 1][1] if i + 1 < len(event_times) else times[-1]))
        for i, (desc, start_time) in enumerate(event_times)
        if desc == event_desc
    ]

    # Handle cases where no or multiple segments are found
    if not segments:
        return "No such segment", None, None, None
    if len(segments) > 1:
        return "Multiple segments found", None, None, None

    # Extract the data corresponding to the identified segment
    segment_mask: np.ndarray = (times >= segments[0][0]) & (times < segments[0][1])
    segment_data: np.ndarray = data[:, segment_mask]
    result, pxx_avg, band_frequencies, ei = process_eeg(pd.DataFrame(segment_data), sampling_freq)

    # Remove artifacts from the segmented data
    f_a_data: np.ndarray = remove_artifacts(segment_data, sampling_freq)

    # Prepare plot data for power spectral density
    psds: List[dict] = [None] * len(bands)
    if pxx_avg is not None and band_frequencies is not None:
        for i in range(len(bands)):
            points: List[Tuple[float, float]] = [(band_frequencies[i][j], pxx_avg[i][j]) for j in range(len(band_frequencies[i]))]
            psds[i] = {"band": bands[i][0], "points": points}

    # Aggregate cleaned data for visualization
    if f_a_data is not None:
        aggregated_cleaned_data: np.ndarray = np.mean(f_a_data, axis=0)
        cleaned_data: List[float] = aggregated_cleaned_data.tolist()
    else:
        cleaned_data: List[float] = []

    return result, psds, cleaned_data, ei
