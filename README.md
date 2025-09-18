# Audio Quality Metrics Calculator

A comprehensive Python tool for calculating ML/AI-based audio quality metrics.

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **For some advanced metrics, you may need additional packages:**
```bash
# Optional: For more advanced speech metrics
pip install speechbrain

# Optional: For transformer-based metrics
pip install sentence-transformers
```

## Quick Start

### Command Line Usage

```bash
# Analyze single audio file
python audio_quality_metrics.py audio.wav

# Compare audio with reference
python audio_quality_metrics.py test_audio.wav --reference clean_audio.wav

# Custom output file and sample rate
python audio_quality_metrics.py audio.wav --output my_results.json --sample-rate 22050
```

### Programmatic Usage

```python
from audio_quality_metrics import AudioQualityAnalyzer

# Single audio analysis
analyzer = AudioQualityAnalyzer("audio.wav")
results = analyzer.analyze_all()
analyzer.save_results("results.json")

# With reference audio
analyzer = AudioQualityAnalyzer("test.wav", reference_path="clean.wav")
results = analyzer.analyze_all()
```

## Supported Metrics

### Basic Audio Features
- **Duration**: Length of the audio signal in seconds
- **RMS Energy**: Root Mean Square energy indicating overall loudness/amplitude
- **Zero-Crossing Rate**: Frequency of signal sign changes, useful for distinguishing voiced/unvoiced speech
- **Spectral Centroid**: The "center of mass" of the spectrum, correlates with brightness perception
- **Spectral Rolloff**: Frequency below which 85% of spectral energy is contained
- **Spectral Bandwidth**: Weighted spectral spread around the centroid
- **MFCC**: Mel-Frequency Cepstral Coefficients, compact representation of spectral envelope

### Perceptual Quality Metrics (requires reference audio)
These metrics predict human perception of audio quality by comparing test audio to clean reference:

- **PESQ (Perceptual Evaluation of Speech Quality)**: ITU-T P.862 standard that predicts speech quality on a 1-5 scale (higher is better). Widely used for VoIP and codec evaluation
  - *Narrowband*: 8kHz sampling, traditional telephony
  - *Wideband*: 16kHz sampling, better for modern communications
- **STOI (Short-Time Objective Intelligibility)**: Measures speech intelligibility on 0-1 scale (1 = perfect intelligibility). Better than PESQ for predicting human speech understanding
- **Extended STOI**: Improved version of STOI with better correlation to human listening tests

### Spectral Distance Metrics
Measure differences between audio signals in the frequency domain:

- **Log-Spectral Distance (LSD)**: Average difference between log-magnitude spectra in dB. Lower values indicate better quality
- **Spectral Convergence**: Normalized Frobenius norm of spectral magnitude difference. Measures how well spectral content is preserved
- **MFCC Distance**: Euclidean distance between MFCC feature vectors. Captures perceptually relevant spectral differences
- **Spectral Flatness**: Measure of how tone-like vs noise-like a sound is (0 = pure tone, 1 = white noise)

### Signal Quality Estimation
Objective measures of signal characteristics:

- **SNR (Signal-to-Noise Ratio)**: Ratio of signal power to noise power in dB. Higher values indicate cleaner audio
- **Dynamic Range**: Difference between loudest and quietest parts in dB. Measures the signal's expressiveness
- **THD (Total Harmonic Distortion)**: Percentage of unwanted harmonics added to the original signal. Lower is better
- **Clipping Detection**: Ratio of samples at maximum amplitude, indicating potential distortion

### Advanced Audio Features
Specialized characteristics for detailed audio analysis:

- **Harmonic-Percussive Separation**: Separates tonal (harmonic) and transient (percussive) components
  - *Harmonic Ratio*: Proportion of harmonic content (melodic instruments, voices)
  - *Percussive Ratio*: Proportion of percussive content (drums, attacks)
- **Tempo**: Estimated beats per minute from rhythm analysis
- **Chroma Features**: 12-dimensional pitch class profile representing musical notes (C, C#, D, etc.)
- **Spectral Contrast**: Difference between peaks and valleys in spectral subbands, measures spectral complexity

## Metric Interpretation Guide

Understanding what the values mean for practical audio quality assessment:

### Quality Score Ranges

**PESQ Scores:**
- 4.0-4.5: Excellent quality (HD audio)
- 3.5-4.0: Good quality (standard phone calls)
- 3.0-3.5: Fair quality (compressed audio)
- 2.5-3.0: Poor quality (heavily compressed)
- 1.0-2.5: Bad quality (artifacts present)

**STOI Scores:**
- 0.9-1.0: Excellent intelligibility
- 0.8-0.9: Good intelligibility
- 0.7-0.8: Acceptable intelligibility
- 0.6-0.7: Poor intelligibility
- <0.6: Very poor intelligibility

**SNR Values:**
- \>30 dB: Excellent (studio quality)
- 20-30 dB: Good (clean recording)
- 15-20 dB: Acceptable (some background noise)
- 10-15 dB: Poor (noticeable noise)
- <10 dB: Very poor (noise dominates)

**Dynamic Range:**
- \>60 dB: Excellent (professional recording)
- 40-60 dB: Good (consumer audio)
- 20-40 dB: Compressed audio
- <20 dB: Heavily compressed/limited

### Use Cases by Application

**Speech Quality Assessment:**
- Primary: PESQ, STOI, SNR
- Secondary: Spectral centroid, MFCC distance

**Music Quality Assessment:**
- Primary: Dynamic range, THD, spectral convergence
- Secondary: Harmonic ratio, tempo, chroma features

**Audio Codec Evaluation:**
- Primary: Log-spectral distance, MFCC distance
- Secondary: Spectral flatness, clipping detection

**Noise Reduction Testing:**
- Primary: SNR, STOI (if reference available)
- Secondary: Spectral centroid changes, RMS energy

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "basic_metrics": {
    "duration_seconds": 3.0,
    "rms_energy_mean": 0.1234,
    "spectral_centroid_mean": 2500.0,
    "mfcc_mean": [1.2, 3.4, ...]
  },
  "pesq_stoi": {
    "pesq_wideband": 2.5,
    "stoi": 0.85,
    "extended_stoi": 0.87
  },
  "signal_quality": {
    "snr_estimate_db": 25.3,
    "dynamic_range_db": 45.2,
    "clipping_ratio": 0.001
  },
  "metadata": {
    "audio_file": "test.wav",
    "analysis_timestamp": "2025-09-18T10:30:00",
    "sample_rate": 16000
  }
}
```

## Troubleshooting

### Common Issues

1. **"pesq and pystoi packages not installed"**
   - Install with: `pip install pesq pystoi`

2. **"Reference audio required"**
   - Some metrics (PESQ, STOI, spectral distances) need a clean reference
   - Provide reference with `--reference` or `reference_path` parameter

3. **Sample rate issues**
   - PESQ requires specific sample rates (8kHz for narrowband, 16kHz for wideband)
   - The script handles resampling automatically

4. **Memory issues with large files**
   - Consider processing audio in chunks for very long files
   - Reduce sample rate if high frequency content isn't critical

### Performance Tips

- Use 16kHz sample rate for speech analysis (default)
- Use 22kHz+ for music analysis
- Provide reference audio for comparative metrics
- Check that audio files are not corrupted

## Example Demo

Run the example with generated sample audio:

```bash
python example_usage.py
```

This will:
1. Create sample clean and noisy audio files
2. Run all available metrics
3. Save results to `demo_metrics.json`
4. Print a summary of key metrics

## Extending the Tool

To add new metrics:

1. Add a new method to `AudioQualityAnalyzer` class
2. Call it from `analyze_all()` method
3. Handle errors gracefully with try/except

Example:
```python
def calculate_new_metric(self):
    try:
        # Your metric calculation here
        result = some_metric_function(self.audio)
        self.results['new_metric'] = {'value': float(result)}
    except Exception as e:
        self.results['new_metric'] = {'error': str(e)}
```
