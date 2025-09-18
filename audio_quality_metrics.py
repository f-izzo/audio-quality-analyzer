#!/usr/bin/env python3
"""
Audio Quality Metrics Calculator
Calculates various ML/AI-based audio quality metrics for recorded audio files.

Requirements:
pip install librosa soundfile numpy scipy scikit-learn torch torchaudio pesq pystoi speechmetrics transformers

Note: Some metrics require reference audio or specific model downloads.
"""

import os
import json
import warnings
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

class AudioQualityAnalyzer:
    def __init__(self, audio_path, reference_path=None, sample_rate=16000):
        """
        Initialize the audio quality analyzer.
        
        Args:
            audio_path: Path to the audio file to analyze
            reference_path: Path to reference audio (optional, needed for some metrics)
            sample_rate: Target sample rate for processing
        """
        self.audio_path = audio_path
        self.reference_path = reference_path
        self.sample_rate = sample_rate
        self.results = {}
        
        # Load audio
        self.audio, self.sr = librosa.load(audio_path, sr=sample_rate)
        if reference_path:
            self.reference, _ = librosa.load(reference_path, sr=sample_rate)
        else:
            self.reference = None
            
    def calculate_basic_metrics(self):
        """Calculate basic audio metrics"""
        try:
            # Duration
            duration = len(self.audio) / self.sr
            
            # RMS Energy
            rms = librosa.feature.rms(y=self.audio)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(self.audio)[0]
            zcr_mean = np.mean(zcr)
            
            # Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio, sr=self.sr)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            self.results['basic_metrics'] = {
                'duration_seconds': float(duration),
                'rms_energy_mean': float(rms_mean),
                'rms_energy_std': float(rms_std),
                'zero_crossing_rate_mean': float(zcr_mean),
                'spectral_centroid_mean': float(spectral_centroid_mean),
                'spectral_rolloff_mean': float(spectral_rolloff_mean),
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist()
            }
            
        except Exception as e:
            self.results['basic_metrics'] = {'error': str(e)}
    
    def calculate_pesq_stoi(self):
        """Calculate PESQ and STOI metrics (requires reference audio)"""
        if self.reference is None:
            self.results['pesq_stoi'] = {'error': 'Reference audio required for PESQ and STOI'}
            return
            
        try:
            from pesq import pesq
            from pystoi import stoi
            
            # Ensure same length
            min_len = min(len(self.audio), len(self.reference))
            audio_trimmed = self.audio[:min_len]
            ref_trimmed = self.reference[:min_len]
            
            # PESQ (narrowband and wideband)
            try:
                pesq_nb = pesq(self.sr, ref_trimmed, audio_trimmed, 'nb')
                pesq_wb = pesq(self.sr, ref_trimmed, audio_trimmed, 'wb')
            except:
                # Try with 8kHz for narrowband
                audio_8k = librosa.resample(audio_trimmed, orig_sr=self.sr, target_sr=8000)
                ref_8k = librosa.resample(ref_trimmed, orig_sr=self.sr, target_sr=8000)
                pesq_nb = pesq(8000, ref_8k, audio_8k, 'nb')
                pesq_wb = None
            
            # STOI
            stoi_score = stoi(ref_trimmed, audio_trimmed, self.sr, extended=False)
            estoi_score = stoi(ref_trimmed, audio_trimmed, self.sr, extended=True)
            
            self.results['pesq_stoi'] = {
                'pesq_narrowband': float(pesq_nb) if pesq_nb else None,
                'pesq_wideband': float(pesq_wb) if pesq_wb else None,
                'stoi': float(stoi_score),
                'extended_stoi': float(estoi_score)
            }
            
        except ImportError:
            self.results['pesq_stoi'] = {'error': 'pesq and pystoi packages not installed'}
        except Exception as e:
            self.results['pesq_stoi'] = {'error': str(e)}
    
    def calculate_spectral_distances(self):
        """Calculate spectral distance metrics"""
        try:
            # Get spectrogram
            S = librosa.stft(self.audio)
            magnitude = np.abs(S)
            log_magnitude = np.log(magnitude + 1e-8)
            
            if self.reference is not None:
                # Reference spectrogram
                S_ref = librosa.stft(self.reference)
                mag_ref = np.abs(S_ref)
                log_mag_ref = np.log(mag_ref + 1e-8)
                
                # Align shapes
                min_frames = min(magnitude.shape[1], mag_ref.shape[1])
                magnitude = magnitude[:, :min_frames]
                mag_ref = mag_ref[:, :min_frames]
                log_magnitude = log_magnitude[:, :min_frames]
                log_mag_ref = log_mag_ref[:, :min_frames]
                
                # Log-Spectral Distance
                lsd = np.mean(np.sqrt(np.mean((log_magnitude - log_mag_ref)**2, axis=0)))
                
                # Spectral Convergence
                spectral_conv = np.linalg.norm(magnitude - mag_ref, 'fro') / np.linalg.norm(mag_ref, 'fro')
                
                # MFCC Distance
                mfcc_orig = librosa.feature.mfcc(y=self.audio[:min_frames*512], sr=self.sr, n_mfcc=13)
                mfcc_ref = librosa.feature.mfcc(y=self.reference[:min_frames*512], sr=self.sr, n_mfcc=13)
                mfcc_distance = np.mean(np.sqrt(np.sum((mfcc_orig - mfcc_ref)**2, axis=0)))
                
                self.results['spectral_distances'] = {
                    'log_spectral_distance': float(lsd),
                    'spectral_convergence': float(spectral_conv),
                    'mfcc_distance': float(mfcc_distance)
                }
            else:
                # Calculate spectral statistics without reference
                spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=self.audio))
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=self.audio, sr=self.sr))
                
                self.results['spectral_distances'] = {
                    'spectral_flatness': float(spectral_flatness),
                    'spectral_bandwidth': float(spectral_bandwidth),
                    'note': 'Reference audio needed for distance metrics'
                }
                
        except Exception as e:
            self.results['spectral_distances'] = {'error': str(e)}
    
    def calculate_advanced_metrics(self):
        """Calculate advanced ML-based metrics (requires additional models)"""
        try:
            # Harmonic-Percussive Separation
            y_harmonic, y_percussive = librosa.effects.hpss(self.audio)
            harmonic_ratio = np.mean(librosa.feature.rms(y=y_harmonic)[0])
            percussive_ratio = np.mean(librosa.feature.rms(y=y_percussive)[0])
            
            # Tempo
            tempo, beats = librosa.beat.beat_track(y=self.audio, sr=self.sr)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=self.audio, sr=self.sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr)
            contrast_mean = np.mean(contrast, axis=1)
            
            self.results['advanced_metrics'] = {
                'harmonic_ratio': float(harmonic_ratio),
                'percussive_ratio': float(percussive_ratio),
                'tempo': float(tempo),
                'chroma_mean': chroma_mean.tolist(),
                'chroma_std': chroma_std.tolist(),
                'spectral_contrast_mean': contrast_mean.tolist()
            }
            
        except Exception as e:
            self.results['advanced_metrics'] = {'error': str(e)}
    
    def calculate_signal_quality_metrics(self):
        """Calculate signal quality metrics"""
        try:
            # Signal-to-Noise Ratio estimation (simple approach)
            # Estimate noise as the quietest 10% of the signal
            sorted_audio = np.sort(np.abs(self.audio))
            noise_level = np.mean(sorted_audio[:int(0.1 * len(sorted_audio))])
            signal_level = np.mean(sorted_audio[int(0.9 * len(sorted_audio)):])
            snr_estimate = 20 * np.log10(signal_level / (noise_level + 1e-8))
            
            # Dynamic range
            max_amplitude = np.max(np.abs(self.audio))
            min_amplitude = np.min(np.abs(self.audio[self.audio != 0])) if len(self.audio[self.audio != 0]) > 0 else 1e-8
            dynamic_range = 20 * np.log10(max_amplitude / min_amplitude)
            
            # Clipping detection
            clipping_ratio = np.sum(np.abs(self.audio) > 0.99) / len(self.audio)
            
            # THD estimation (simplified)
            fft = np.fft.fft(self.audio)
            freqs = np.fft.fftfreq(len(self.audio), 1/self.sr)
            magnitude = np.abs(fft)
            
            # Find fundamental frequency
            fundamental_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            fundamental_freq = freqs[fundamental_idx]
            
            # Estimate harmonics (simplified)
            harmonic_power = 0
            fundamental_power = magnitude[fundamental_idx]**2
            
            for i in range(2, 6):  # 2nd to 5th harmonics
                harmonic_freq = i * fundamental_freq
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                if harmonic_idx < len(magnitude):
                    harmonic_power += magnitude[harmonic_idx]**2
            
            thd_estimate = np.sqrt(harmonic_power) / np.sqrt(fundamental_power) if fundamental_power > 0 else 0
            
            self.results['signal_quality'] = {
                'snr_estimate_db': float(snr_estimate),
                'dynamic_range_db': float(dynamic_range),
                'clipping_ratio': float(clipping_ratio),
                'thd_estimate': float(thd_estimate),
                'max_amplitude': float(max_amplitude),
                'fundamental_frequency': float(fundamental_freq)
            }
            
        except Exception as e:
            self.results['signal_quality'] = {'error': str(e)}
    
    def analyze_all(self):
        """Run all available analyses"""
        print("Calculating basic metrics...")
        self.calculate_basic_metrics()
        
        print("Calculating PESQ and STOI...")
        self.calculate_pesq_stoi()
        
        print("Calculating spectral distances...")
        self.calculate_spectral_distances()
        
        print("Calculating advanced metrics...")
        self.calculate_advanced_metrics()
        
        print("Calculating signal quality metrics...")
        self.calculate_signal_quality_metrics()
        
        # Add metadata
        self.results['metadata'] = {
            'audio_file': self.audio_path,
            'reference_file': self.reference_path,
            'sample_rate': self.sample_rate,
            'analysis_timestamp': datetime.now().isoformat(),
            'audio_length_samples': len(self.audio),
            'audio_duration_seconds': len(self.audio) / self.sr
        }
        
        return self.results
    
    def save_results(self, output_path):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate audio quality metrics')
    parser.add_argument('audio_file', help='Path to audio file to analyze')
    parser.add_argument('--reference', help='Path to reference audio file (optional)')
    parser.add_argument('--output', default='audio_metrics.json', help='Output JSON file')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate for processing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file {args.audio_file} not found")
        return
    
    if args.reference and not os.path.exists(args.reference):
        print(f"Error: Reference file {args.reference} not found")
        return
    
    # Run analysis
    analyzer = AudioQualityAnalyzer(
        audio_path=args.audio_file,
        reference_path=args.reference,
        sample_rate=args.sample_rate
    )
    
    results = analyzer.analyze_all()
    analyzer.save_results(args.output)
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")

    def _format_numeric(value, fmt):
        """Return formatted value when numeric, otherwise pass through as string"""
        if isinstance(value, (int, float)):
            return format(value, fmt)
        if value is None:
            return "N/A"
        return str(value)

    if 'basic_metrics' in results:
        duration = results['basic_metrics'].get('duration_seconds')
        print(f"Duration: {_format_numeric(duration, '.2f')} seconds")

        rms = results['basic_metrics'].get('rms_energy_mean')
        print(
            "RMS Energy: "
            f"{_format_numeric(rms, '.4f')} (expected 0.0-1.0 for normalized audio)"
        )

    if 'signal_quality' in results:
        snr = results['signal_quality'].get('snr_estimate_db')
        print(
            "SNR Estimate: "
            f"{_format_numeric(snr, '.2f')} dB (typical speech sits around 10-30 dB)"
        )

        dynamic_range = results['signal_quality'].get('dynamic_range_db')
        print(
            "Dynamic Range: "
            f"{_format_numeric(dynamic_range, '.2f')} dB (0-120 dB, higher means more headroom)"
        )

        clipping_ratio = results['signal_quality'].get('clipping_ratio')
        print(
            "Clipping Ratio: "
            f"{_format_numeric(clipping_ratio, '.4f')} (0.0 means no clipping, 1.0 is always clipped)"
        )

    if 'pesq_stoi' in results and 'error' not in results['pesq_stoi']:
        pesq_score = results['pesq_stoi'].get('pesq_wideband')
        print(
            "PESQ: "
            f"{_format_numeric(pesq_score, '.2f')} (wideband scores typically range 1.0-4.5)"
        )

        stoi_score = results['pesq_stoi'].get('stoi')
        print(
            "STOI: "
            f"{_format_numeric(stoi_score, '.4f')} (0.0-1.0, with higher being more intelligible)"
        )

if __name__ == "__main__":
    main()
