# 10 - Speech Emotion Recognition (SER) Application

## Overview
This application is part of the **Text and Speech Analysis** college lab curriculum. It demonstrates **Acoustic Signal Processing and Speech Emotion Recognition (SER)**, the task of detecting human emotional states directly from acoustic sound waves and speech prosody.

---

## Theory & Acoustic Features
In speech signal processing, emotion modulates human vocal tract acoustics through physiological arousal (e.g., increased respiratory pressure, vocal fold tension):

1. **Root Mean Square (RMS) Energy (Signal Power)**:
   Measures perceptual volume/loudness:
   $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x[i]^2}$$
   *High arousal states (Angry, Happy) exhibit high RMS energy; Sadness exhibits low energy.*

2. **Zero-Crossing Rate (ZCR)**:
   The frequency at which the audio waveform crosses the zero amplitude axis:
   $$\text{ZCR} = \frac{1}{2(N-1)}\sum_{i=1}^{N-1} |\operatorname{sgn}(x[i]) - \operatorname{sgn}(x[i-1])|$$
   *Indicates spectral noisiness and high-frequency content.*

3. **Dominant Pitch Frequency ($F_0$ Fundamental)**:
   Determined via Fast Fourier Transform (FFT) peak identification across the human vocal frequency spectrum (80 Hz - 3500 Hz).
   *Angry/Happy voices show marked pitch elevation ($>250\text{ Hz}$); Sad/Neutral speech shows lower, flatter fundamental pitch ($<160\text{ Hz}$).*

4. **Spectral Centroid (Timbre / Brightness)**:
   The "center of mass" of the frequency spectrum:
   $$\text{Centroid} = \frac{\sum_k f[k] \cdot |X[k]|}{\sum_k |X[k]|}$$
   *Higher centroids correlate with sharper, tenser vocal quality.*

---

## Features
- **Pure Scientific Python**: Built using standard `numpy` and `scipy` — avoids difficult-to-compile heavy C++ audio libraries.
- **Audio Synthesizer Helper**: Includes `generate_sample_audio.py` which mathematically generates synthetic benchmark audio files (`sample_angry.wav`, `sample_happy.wav`, `sample_sad.wav`, `sample_neutral.wav`) so students can run experiments immediately without recording equipment.
- **Detailed Acoustic Report**: Prints exact physical measurements (RMS energy, ZCR, pitch frequency in Hz, spectral centroid, energy variance) alongside emotional probability distributions.
- **WAV Compatibility**: Analyzes any mono or stereo 16-bit PCM `.wav` audio recording.

---

## Prerequisites & Installation

1. Navigate to this folder:
   ```bash
   cd 10_Speech_Emotion_Recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1: (Optional) Generate Benchmark Test Audio
```bash
python generate_sample_audio.py
```
This generates `sample_angry.wav`, `sample_happy.wav`, `sample_sad.wav`, and `sample_neutral.wav`.

### Step 2: Run the Classifier Application
```bash
python speech_emotion_recognizer.py
```

### Example Usage:
```text
=================================================================
           SPEECH EMOTION RECOGNITION REPORT
=================================================================
File Analyzed      : sample_angry.wav
Duration           : 2.50 seconds
Sampling Rate      : 16000 Hz
-----------------------------------------------------------------
Acoustic Feature Extraction:
  • RMS Energy (Loudness)        : 0.5841
  • Zero-Crossing Rate (ZCR)     : 0.0410
  • Dominant Pitch Frequency     : 320.0 Hz
  • Spectral Centroid (Timbre)   : 785.4 Hz
  • Frame Energy Variance        : 0.0821
-----------------------------------------------------------------
PREDICTED EMOTION  : ANGRY 😡
Classification Confidence : 42.1%
-----------------------------------------------------------------
Emotion Probability Distribution:
  Angry 😡     : [########------------]  42.1%
  Happy 😄     : [######--------------]  30.8%
  Neutral 😐   : [###-----------------]  16.2%
  Sad 😢       : [##------------------]  10.9%
=================================================================
```

---

## Technical Notes & Limitations
- **Supported Format**: Uncompressed PCM `.wav` files (8 kHz to 48 kHz). MP3 or AAC files should be converted to WAV before running.
- **Prosodic vs Semantic**: This system evaluates **acoustic speech prosody** (how words sound physically), not lexical transcripts (what words mean).
