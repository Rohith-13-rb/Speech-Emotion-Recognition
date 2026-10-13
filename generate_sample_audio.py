"""
Sample Audio Generator for Speech Emotion Recognition
Subject: Text and Speech Analysis
Description: Synthesizes synthetic WAV audio files exhibiting emotional acoustic
             prosody profiles (Angry, Happy, Sad, Neutral) using pure scipy/numpy.
"""

import math
import os
import sys

try:
    import numpy as np
    from scipy.io import wavfile
except ImportError:
    print("\n[ERROR] 'numpy' or 'scipy' is not installed.")
    print("Please run: pip install -r requirements.txt\n")
    sys.exit(1)


def generate_tone_sequence(duration=2.5, sr=16000, profile="neutral"):
    """
    Generates synthesized sound samples modeling emotional prosodic characteristics:
    - Angry: High energy (loud), elevated pitch, rapid frequency jitter/modulation.
    - Happy: Moderately high energy, cheerful fluctuating pitch, upward lilt.
    - Sad: Low energy (soft), low fundamental pitch, slow decay, descending glide.
    - Neutral: Moderate energy, steady monotonous pitch.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if profile == "angry":
        # High base frequency ~350Hz with intense vibrato/jitter and loud amplitude
        freq = 320.0 + 80.0 * np.sin(2 * np.pi * 8 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        signal = 0.85 * np.sin(phase) + 0.3 * np.sin(2 * phase)
        # Add harsh bursts
        envelope = np.ones_like(t)

    elif profile == "happy":
        # Elevated cheerful pitch ~260-340Hz with bouncy modulation
        freq = 280.0 + 60.0 * np.sin(2 * np.pi * 3.5 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        signal = 0.65 * np.sin(phase) + 0.2 * np.sin(2 * phase)
        # Bouncy bell envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)

    elif profile == "sad":
        # Low, subdued pitch ~120-140Hz, descending pitch glide, quiet amplitude
        freq = np.linspace(150.0, 110.0, len(t))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        signal = 0.25 * np.sin(phase)
        # Soft slow decay
        envelope = np.exp(-1.0 * t / duration)

    else:  # neutral
        # Flat, steady pitch ~180Hz, moderate amplitude
        freq = 180.0 * np.ones_like(t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        signal = 0.45 * np.sin(phase)
        envelope = np.ones_like(t) * 0.8

    composite = signal * envelope
    # Normalize to 16-bit PCM range
    max_val = np.max(np.abs(composite))
    if max_val > 0:
        composite = composite / max_val * 0.9
    int16_audio = (composite * 32767).astype(np.int16)
    return sr, int16_audio


def main():
    print("=" * 65)
    print("      SAMPLE AUDIO GENERATOR - SPEECH EMOTION RECOGNITION")
    print("=" * 65)
    print("Synthesizing benchmark WAV audio files with distinct prosody profiles...\n")

    emotions = ["angry", "happy", "sad", "neutral"]
    for emo in emotions:
        filename = f"sample_{emo}.wav"
        sr, audio = generate_tone_sequence(duration=2.5, profile=emo)
        wavfile.write(filename, sr, audio)
        print(f"[CREATED] {filename:<18} (Sample Rate: {sr}Hz, Duration: 2.5s, Profile: {emo.upper()})")

    print("\n[SUCCESS] 4 sample audio files generated in current directory.")
    print("You can now run 'python speech_emotion_recognizer.py' to analyze them!\n")


if __name__ == "__main__":
    main()
