import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from scipy.signal import butter, filtfilt, iirnotch, welch # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import mne # type: ignore
from networkx import is_path # type: ignore

bands = [
            ('Delta', 0.5, 4),  # index = 0
            ('Theta', 4, 8),    # index = 1
            ('Alpha', 8, 12),   # index = 2
            ('Beta', 12, 30),   # index = 3
            ('Gamma', 30, 50)   # index = 4
        ]

# Load the pre-trained model
MODEL_PATH = "/home/dhruba/projects/EEG-analysis/EEG-analysis-backend/binary_classification.pkl"
model = joblib.load(MODEL_PATH)

 
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    if data.ndim == 1:
        filtered_data = filtfilt(b, a, data)
    else:
        filtered_data = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=1, arr=data)
    return filtered_data

def remove_artifacts(eeg_data, fs, bandpass=(1, 50), notch_freq=50, notch_quality=30):
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def apply_notch_filter(data, freq, fs, quality):
        nyquist = 0.5 * fs
        w0 = freq / nyquist
        b, a = iirnotch(w0, quality)
        return filtfilt(b, a, data)

    lowcut, highcut = bandpass
    filtered_data = np.apply_along_axis(lambda x: bandpass_filter(x, lowcut, highcut, fs), axis=1, arr=eeg_data)
    cleaned_data = np.apply_along_axis(lambda x: apply_notch_filter(x, notch_freq, fs, notch_quality), axis=1, arr=filtered_data)
    # print(cleaned_data)
    return cleaned_data

def process_eeg_data(data, sampling_frequency, window_duration):
    samples_per_window = int(sampling_frequency * window_duration)
    all_windows = []
    if data.ndim != 2:
        raise ValueError(f"Data must be a 2D array (channels x samples).")

    num_channels, total_samples = data.shape
    num_windows = total_samples // samples_per_window
    trimmed_data = data[:, :num_windows * samples_per_window]
    windows = trimmed_data.reshape(num_channels, num_windows, samples_per_window).transpose(1, 0, 2)
    all_windows.append(windows)
    combined_windows = np.concatenate(all_windows, axis=0)
    return combined_windows

def compute_power_features(eeg_data, fs):
    power_features = []
    band_freuencies : list[list[np.float64]] = [None] * len(bands)
    f, _ = welch(eeg_data[0][0], fs, nperseg=1024)
    for i in range(len(bands)):
        mask = (f >= bands[i][1]) & (f <= bands[i][2])
        band_freuencies[i] = f[mask]
    
    pxx_avg : list[list[np.float64]] = [[0] * len(b) for b in band_freuencies]

    for window in eeg_data:
        power_window = []
        for channel_data in window:
            _, Pxx = welch(channel_data, fs, nperseg=1024)
            power_band = []
            for i, (_, f_min, f_max) in enumerate(bands):
                band_mask = (f >= f_min) & (f <= f_max)
                curr_Pxx = Pxx[band_mask]
                for j in range(len(band_freuencies[i])):
                    pxx_avg[i][j] += curr_Pxx[j]

                band_power = np.trapz(curr_Pxx, band_freuencies[i])
                power_band.append(band_power)
            power_window.append(power_band)
        power_features.append(power_window)

    for pxx_curr in pxx_avg:
        for i in range(len(pxx_curr)):
            pxx_curr[i] /= (eeg_data.shape[0] * eeg_data.shape[1])

    return np.array(power_features), pxx_avg, band_freuencies

def compute_entropy_features(eeg_data):
    def approximate_entropy(x, m=2, r=0.2):
        N = len(x)
        phi = []
        for m in range(1, m+1):
            X = np.array([x[i:i+m] for i in range(N - m + 1)])
            C = np.array([np.sum(np.abs(X - X[i]) <= r) / (N - m + 1) for i in range(N - m)])
            phi.append(np.mean(np.log(C)))
        return phi[0] - phi[1]

    entropy_features = []
    for window in eeg_data:
        entropy_window = []
        for channel_data in window:
            entropy_window.append(approximate_entropy(channel_data))
        entropy_features.append(entropy_window)
    return np.array(entropy_features)


def calculate_engagement_index(power_features: np.ndarray) -> np.float64:
    """
    Calculate Engagement Index (EI) from the average power spectral densities.
    
    EI = Beta / (Alpha + Theta)
    """
    
    print(power_features.shape[0] * power_features.shape[1])
    if (power_features.shape[0] * power_features.shape[1]) == 0:
        return np.float64(0)
    
    alpha_index : int = 2
    beta_index : int = 3
    theta_index : int = 1

    alpha_power : np.float64 = np.float64(0)
    beta_power : np.float64 = np.float64(0)
    theta_power : np.float64 = np.float64(0)

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
    

def preprocess_eeg_data(eeg_data, fs):
    power_features, pxx_avg, band_frequencies = compute_power_features(eeg_data, fs)
    entropy_features = compute_entropy_features(eeg_data)
    entropy_features = entropy_features[:, :, np.newaxis]
    combined_features = np.concatenate((power_features, entropy_features), axis=-1)
    n_windows, _, _ = combined_features.shape
    combined_features = combined_features.reshape(n_windows, -1)
    scaler = MinMaxScaler()
    standardized_features = scaler.fit_transform(combined_features)
    labels = np.ones((n_windows, 1))
    preprocessed_data = np.hstack((standardized_features, labels))
    return preprocessed_data, pxx_avg, band_frequencies, calculate_engagement_index(power_features)

def check_load(result):
    ones = 0
    zeros = 0
    for x in result:
        if x == np.float64(1.0):
            ones += 1
        else:
            zeros += 1

    print("ones :", ones)
    print("zeros :", zeros)
    return "Load Present" if ones > zeros else "No Load"

def process_eeg(data : pd.DataFrame):
    try:
        lowcut = 1
        highcut = 50
        fs = 512
        window_duration = 2

        filtered_data = bandpass_filter(data, lowcut, highcut, fs)
        f_a_data = remove_artifacts(filtered_data, fs, bandpass=(1, 50), notch_freq=50, notch_quality=30)
        segmented_data = process_eeg_data(f_a_data, fs, window_duration)

        features_data, pxx_avg, band_frequencies, EI = preprocess_eeg_data(segmented_data, fs)
        features_data = pd.DataFrame(features_data)
        selected_features = features_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20,
                                                   21, 22, 26, 28, 29, 30, 31, 32, 33, 35, 38, 39, 41, 47, 50, 52,
                                                   54, 55, 57, 58, 59, 60, 64, 70, 71, 79, 81, 82, 83, 89, 94, 95]]
        result = model.predict(selected_features)
        return check_load(result), pxx_avg, band_frequencies, EI

    except FileNotFoundError:
        print(f"Error: File not found at {is_path}")
        return None, None, None, None


def get_events_from_vmrk(vmrk_file : str) -> list[tuple]:
    events : list[tuple] = []

    with open(vmrk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Mk"):
                info = line.strip().split(",")  # Split line by comma
                # Check if the line has at least 4 elements
                # Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>, <Size in data points>, <Channel number (0 = marker is related to all channels)>
                if (len(info) >= 4):
                    desc = info[1].strip()
                    index = int(info[2].strip())
                    events.append((desc, index))

    return events
    


def process_eeg_file(vhdr_filename: str, vmrk_filename: str, event_desc: str):
    # Load EEG
    raw = mne.io.read_raw_brainvision(vhdr_filename, preload=True)
    sampling_freq = 512
    data, times = raw.get_data(return_times=True)

    # Get events
    events = get_events_from_vmrk(vmrk_filename)
    event_times = [(desc, sample_index / sampling_freq) for desc, sample_index in events]

    # Identify segment
    segments = [
        (start_time, (event_times[i + 1][1] if i + 1 < len(event_times) else times[-1]))
        for i, (desc, start_time) in enumerate(event_times)
        if desc == event_desc
    ]

    if not segments:
        return "No such segment", None, None, None
    if len(segments) > 1:
        return "Multiple segments found", None, None, None

    # Process the segment
    segment_mask = (times >= segments[0][0]) & (times < segments[0][1])
    segment_data = data[:, segment_mask]
    result, pxx_avg, band_frequencies, EI = process_eeg(pd.DataFrame(segment_data))

    # Return the cleaned data for visualization
    f_a_data = remove_artifacts(segment_data, sampling_freq)

    # prepare plot data
    plot_data : list[dict] = [None] * len(bands)
    if pxx_avg != None and band_frequencies != None:
        for i in range(len(bands)):
            points : list[tuple] = [(band_frequencies[i][j], pxx_avg[i][j]) for j in range(len(band_frequencies[i]))]
            plot_data[i] = {"band": bands[i][0], "points": points}

    # Format cleaned data
    if f_a_data is not None:
        # Aggregate cleaned data across channels (e.g., by averaging)
        aggregated_cleaned_data = np.mean(
            f_a_data, axis=0
        )  # Average across channels
        cleaned_data = (
            aggregated_cleaned_data.tolist()
        )  # Directly send as a list
    else:
        cleaned_data = []
        

    return result, plot_data, cleaned_data, EI
