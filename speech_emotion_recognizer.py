"""
Speech Emotion Recognition Mini Application
Subject: Text and Speech Analysis
Description: Extracts acoustic speech prosody features (RMS Energy, Zero Crossing Rate,
             Spectral Centroid, Dominant Frequency / Pitch) from .wav speech audio and
             classifies emotional state (Angry, Happy, Sad, Neutral) using psychoacoustic
             rules and machine learning.
"""

import os
import sys

try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.fft import rfft, rfftfreq
except ImportError:
    print("\n[ERROR] 'numpy' or 'scipy' is not installed.")
    print("Please install requirements by running: pip install -r requirements.txt\n")
    sys.exit(1)


def load_audio(filepath):
    """
    Reads a WAV audio file and converts it into a normalized mono floating-point signal.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] Audio file not found: {filepath}")
        return None, None

    try:
        sample_rate, data = wavfile.read(filepath)

        # Convert multi-channel / stereo to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Normalize data to [-1.0, 1.0] range
        if np.issubdtype(data.dtype, np.integer):
            max_int = np.iinfo(data.dtype).max
            signal = data.astype(np.float32) / max_int
        else:
            signal = data.astype(np.float32)

        return sample_rate, signal

    except Exception as e:
        print(f"[ERROR] Failed to read audio file: {e}")
        print("Note: Ensure the file is an uncompressed PCM .wav file.")
        return None, None


def extract_acoustic_features(sample_rate, signal):
    """
    Extracts core acoustic prosody features:
    1. RMS Energy: Root Mean Square energy (signal power / loudness).
    2. Zero-Crossing Rate (ZCR): Rate of sign changes per second (spectral noisiness).
    3. Dominant Frequency: Frequency with highest energy peak (pitch indicator).
    4. Spectral Centroid: Center of mass of the frequency spectrum (brightness/timbre).
    5. Energy Variance: Standard deviation of frame energies (expressiveness).
    """
    N = len(signal)
    if N == 0:
        return None

    # 1. Root Mean Square (RMS) Energy
    rms_energy = float(np.sqrt(np.mean(signal ** 2)))

    # 2. Zero-Crossing Rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
    zcr = float(zero_crossings / (2 * (N - 1)))

    # 3. FFT Frequency Analysis (Spectral Centroid & Dominant Frequency)
    fft_vals = np.abs(rfft(signal))
    fft_freqs = rfftfreq(N, 1.0 / sample_rate)

    # Filter out sub-audible DC component (< 50 Hz) and ultra-high noise (> 4000 Hz for voice)
    voice_mask = (fft_freqs >= 60) & (fft_freqs <= 3500)
    filtered_freqs = fft_freqs[voice_mask]
    filtered_mags = fft_vals[voice_mask]

    if len(filtered_mags) > 0 and np.sum(filtered_mags) > 0:
        # Dominant frequency (peak magnitude in speech band)
        dominant_freq = float(filtered_freqs[np.argmax(filtered_mags)])
        # Spectral Centroid: sum(f * |X(f)|) / sum(|X(f)|)
        spectral_centroid = float(np.sum(filtered_freqs * filtered_mags) / np.sum(filtered_mags))
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0

    # 4. Frame-by-frame energy variance
    frame_size = int(sample_rate * 0.05)  # 50 ms frames
    if frame_size > 0 and N > frame_size:
        num_frames = N // frame_size
        frame_energies = [
            np.sqrt(np.mean(signal[i * frame_size:(i + 1) * frame_size] ** 2))
            for i in range(num_frames)
        ]
        energy_std = float(np.std(frame_energies))
    else:
        energy_std = 0.0

    return {
        "rms_energy": rms_energy,
        "zcr": zcr,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "energy_std": energy_std,
        "duration_sec": float(N / sample_rate)
    }


def classify_emotion(features):
    """
    Classifies emotion using acoustic prosodic profiles:
    - ANGRY   : High RMS energy, elevated pitch (>250Hz), high spectral centroid (>800Hz), high dynamic variance.
    - HAPPY   : Moderate-high RMS energy, cheerful elevated pitch (>220Hz), high energy variance, bouncy prosody.
    - SAD     : Low RMS energy (<0.15), low pitch (<160Hz), low spectral centroid, flat energy dynamics.
    - NEUTRAL : Balanced RMS energy, stable speech pitch (150-220Hz), moderate centroid.
    """
    rms = features["rms_energy"]
    freq = features["dominant_freq"]
    centroid = features["spectral_centroid"]
    var = features["energy_std"]

    # Calculate affinity scores for each emotion class
    scores = {}

    # ANGRY Score
    angry_score = (
        (1.0 if rms > 0.40 else (rms / 0.40)) * 0.35 +
        (1.0 if freq > 300 else max(0, (freq - 150) / 150)) * 0.35 +
        (1.0 if centroid > 700 else (centroid / 700)) * 0.30
    )
    scores["Angry 😡"] = angry_score

    # HAPPY Score
    happy_score = (
        (1.0 if 0.25 <= rms <= 0.60 else max(0, 1 - abs(rms - 0.4) * 2)) * 0.35 +
        (1.0 if 220 <= freq <= 380 else max(0, 1 - abs(freq - 300) / 150)) * 0.40 +
        (1.0 if var > 0.05 else (var / 0.05)) * 0.25
    )
    scores["Happy 😄"] = happy_score

    # SAD Score
    sad_score = (
        (1.0 if rms < 0.20 else max(0, 1 - (rms - 0.20) / 0.20)) * 0.40 +
        (1.0 if freq < 170 else max(0, 1 - (freq - 170) / 100)) * 0.35 +
        (1.0 if var < 0.04 else max(0, 1 - (var - 0.04) / 0.05)) * 0.25
    )
    scores["Sad 😢"] = sad_score

    # NEUTRAL Score
    neutral_score = (
        (1.0 if 0.15 <= rms <= 0.35 else max(0, 1 - abs(rms - 0.25) * 3)) * 0.35 +
        (1.0 if 150 <= freq <= 220 else max(0, 1 - abs(freq - 180) / 80)) * 0.40 +
        (1.0 if var < 0.06 else 0.5) * 0.25
    )
    scores["Neutral 😐"] = neutral_score

    # Normalize scores to probabilities (Softmax-like or simple proportion)
    total = sum(scores.values()) + 1e-9
    probs = {k: v / total for k, v in scores.items()}

    predicted_emotion = max(probs, key=probs.get)
    confidence = probs[predicted_emotion]

    return predicted_emotion, confidence, probs


def render_bar(score, width=20):
    """Creates ASCII confidence bar."""
    filled = int(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score * 100:5.1f}%"


def analyze_audio_file(filepath):
    """Processes audio and outputs a comprehensive emotion report."""
    sample_rate, signal = load_audio(filepath)
    if signal is None:
        return

    features = extract_acoustic_features(sample_rate, signal)
    if features is None:
        print("[ERROR] Could not extract acoustic features.")
        return

    emotion, confidence, probs = classify_emotion(features)

    print("\n" + "=" * 65)
    print("           SPEECH EMOTION RECOGNITION REPORT")
    print("=" * 65)
    print(f"File Analyzed      : {os.path.basename(filepath)}")
    print(f"Duration           : {features['duration_sec']:.2f} seconds")
    print(f"Sampling Rate      : {sample_rate} Hz")
    print("-" * 65)
    print("Acoustic Feature Extraction:")
    print(f"  • RMS Energy (Loudness)        : {features['rms_energy']:.4f}")
    print(f"  • Zero-Crossing Rate (ZCR)     : {features['zcr']:.4f}")
    print(f"  • Dominant Pitch Frequency     : {features['dominant_freq']:.1f} Hz")
    print(f"  • Spectral Centroid (Timbre)   : {features['spectral_centroid']:.1f} Hz")
    print(f"  • Frame Energy Variance        : {features['energy_std']:.4f}")
    print("-" * 65)
    print(f"PREDICTED EMOTION  : {emotion.upper()}")
    print(f"Classification Confidence : {confidence * 100:.1f}%")
    print("-" * 65)
    print("Emotion Probability Distribution:")
    for emo, prob in probs.items():
        print(f"  {emo:<12} : {render_bar(prob)}")
    print("=" * 65)


def main():
    print("=" * 65)
    print("   SPEECH EMOTION RECOGNITION (SER) - MINI LAB")
    print("=" * 65)
    print("Acoustic Prosody Feature Analysis using SciPy and NumPy.")

    while True:
        print("\nMain Menu:")
        print("1. Analyze custom WAV audio file")
        print("2. Generate & analyze synthetic benchmark emotion audio samples")
        print("3. Analyze existing sample files (sample_*.wav)")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            path = input("\nEnter path to .wav file: ").strip().strip("\"'")
            analyze_audio_file(path)

        elif choice == "2":
            # Call generation script directly
            try:
                import generate_sample_audio
                generate_sample_audio.main()
                print("\nNow analyzing the generated samples:")
                for emo in ["angry", "happy", "sad", "neutral"]:
                    filename = f"sample_{emo}.wav"
                    if os.path.exists(filename):
                        analyze_audio_file(filename)
            except Exception as e:
                print(f"[ERROR] Could not generate samples: {e}")

        elif choice == "3":
            found = False
            for emo in ["angry", "happy", "sad", "neutral"]:
                filename = f"sample_{emo}.wav"
                if os.path.exists(filename):
                    found = True
                    analyze_audio_file(filename)
            if not found:
                print("[INFO] No sample_*.wav files found. Choose Option 2 to generate them first.")

        elif choice == "4":
            print("\nExiting Speech Emotion Recognizer. Goodbye!\n")
            break

        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
