#!/usr/bin/env python3
"""
Example usage of the Audio Quality Metrics Calculator

This script demonstrates how to use the AudioQualityAnalyzer class
both programmatically and from command line.
"""

from audio_quality_metrics import AudioQualityAnalyzer
import json
import argparse
import sys

def example_usage():
    """Example of programmatic usage"""
    
    # Example 1: Analyze single audio file (no reference)
    print("=== Example 1: Single Audio Analysis ===")
    
    # Replace with your audio file path
    audio_file = "data/1_hindi.m4a"  # Change this to your audio file
    
    try:
        analyzer = AudioQualityAnalyzer(
            audio_path=audio_file,
            sample_rate=16000  # Adjust based on your audio
        )
        
        results = analyzer.analyze_all()
        
        # Save results
        analyzer.save_results("single_audio_metrics.json")
        
        # Print some key metrics
        print(f"Duration: {results['metadata']['audio_duration_seconds']:.2f} seconds")
        if 'basic_metrics' in results:
            print(f"RMS Energy: {results['basic_metrics']['rms_energy_mean']:.4f}")
        if 'signal_quality' in results:
            print(f"SNR Estimate: {results['signal_quality']['snr_estimate_db']:.2f} dB")
            
    except Exception as e:
        print(f"Error analyzing audio: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Compare audio with reference
    print("=== Example 2: Audio vs Reference Analysis ===")
    
    # Replace with your file paths
    test_audio = "data/1_hindi.m4a"      # Your test audio
    reference_audio = "data/1_marathi.m4a"  # Your reference audio
    
    try:
        analyzer = AudioQualityAnalyzer(
            audio_path=test_audio,
            reference_path=reference_audio,
            sample_rate=16000
        )
        
        results = analyzer.analyze_all()
        analyzer.save_results("comparison_metrics.json")
        
        # Print comparison metrics
        if 'pesq_stoi' in results and 'error' not in results['pesq_stoi']:
            print(f"PESQ Score: {results['pesq_stoi'].get('pesq_wideband', 'N/A')}")
            print(f"STOI Score: {results['pesq_stoi'].get('stoi', 'N/A'):.4f}")
        
        if 'spectral_distances' in results:
            if 'log_spectral_distance' in results['spectral_distances']:
                print(f"Log-Spectral Distance: {results['spectral_distances']['log_spectral_distance']:.4f}")
            
    except Exception as e:
        print(f"Error in comparison analysis: {e}")

def create_sample_audio():
    """Create a sample audio file for testing"""
    import numpy as np
    import soundfile as sf
    
    # Generate a simple test tone
    duration = 3.0  # seconds
    sample_rate = 16000
    frequency = 440  # Hz (A4 note)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a sine wave with some noise
    clean_signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    noise = 0.05 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    # Save both clean and noisy versions
    sf.write("clean_sample.wav", clean_signal, sample_rate)
    sf.write("noisy_sample.wav", noisy_signal, sample_rate)
    
    print("Created sample audio files:")
    print("- clean_sample.wav (reference)")
    print("- noisy_sample.wav (test audio)")
    
    return "noisy_sample.wav", "clean_sample.wav"

def demo_with_sample_audio():
    """Run demo with generated sample audio"""
    print("=== Creating Sample Audio for Demo ===")
    test_audio, reference_audio = create_sample_audio()
    
    print("\n=== Running Analysis on Sample Audio ===")
    analyzer = AudioQualityAnalyzer(
        audio_path=test_audio,
        reference_path=reference_audio,
        sample_rate=16000
    )
    
    results = analyzer.analyze_all()
    analyzer.save_results("demo_metrics.json")
    
    # Print interesting results
    print("\n=== Results Summary ===")
    print(f"Duration: {results['metadata']['audio_duration_seconds']:.2f} seconds")
    
    if 'basic_metrics' in results:
        basic = results['basic_metrics']
        print(f"RMS Energy: {basic['rms_energy_mean']:.4f}")
        print(f"Spectral Centroid: {basic['spectral_centroid_mean']:.2f} Hz")
    
    if 'signal_quality' in results:
        quality = results['signal_quality']
        print(f"SNR Estimate: {quality['snr_estimate_db']:.2f} dB")
        print(f"Dynamic Range: {quality['dynamic_range_db']:.2f} dB")
    
    if 'pesq_stoi' in results and 'error' not in results['pesq_stoi']:
        perceptual = results['pesq_stoi']
        print(f"STOI Score: {perceptual['stoi']:.4f}")
        if perceptual['pesq_wideband']:
            print(f"PESQ Score: {perceptual['pesq_wideband']:.2f}")
    
    if 'spectral_distances' in results:
        spectral = results['spectral_distances']
        if 'log_spectral_distance' in spectral:
            print(f"Log-Spectral Distance: {spectral['log_spectral_distance']:.4f}")

def run_cli_analysis(args):
    """Run analysis based on command line arguments"""
    try:
        analyzer = AudioQualityAnalyzer(
            audio_path=args.test,
            reference_path=args.reference,
            sample_rate=16000
        )
        
        results = analyzer.analyze_all()
        
        # Determine output filename
        if args.reference:
            # Comparison analysis
            output_file = args.output_compare or "comparison_metrics.json"
            print("=== Audio vs Reference Analysis ===")
            
            # Print comparison metrics
            if 'pesq_stoi' in results and 'error' not in results['pesq_stoi']:
                print(f"PESQ Score: {results['pesq_stoi'].get('pesq_wideband', 'N/A')}")
                print(f"STOI Score: {results['pesq_stoi'].get('stoi', 'N/A'):.4f}")
            
            if 'spectral_distances' in results:
                if 'log_spectral_distance' in results['spectral_distances']:
                    print(f"Log-Spectral Distance: {results['spectral_distances']['log_spectral_distance']:.4f}")
        else:
            # Single audio analysis
            output_file = args.output_single or "single_audio_metrics.json"
            print("=== Single Audio Analysis ===")
            
            # Print key metrics
            print(f"Duration: {results['metadata']['audio_duration_seconds']:.2f} seconds")
            if 'basic_metrics' in results:
                print(f"RMS Energy: {results['basic_metrics']['rms_energy_mean']:.4f}")
            if 'signal_quality' in results:
                print(f"SNR Estimate: {results['signal_quality']['snr_estimate_db']:.2f} dB")
        
        # Use --output flag if provided, otherwise use default
        if args.output:
            output_file = args.output
        
        analyzer.save_results(output_file)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        sys.exit(1)

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Audio Quality Metrics Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test audio.wav --output-single single_results.json
  %(prog)s --test audio.wav --reference clean.wav --output-compare comparison.json
  %(prog)s --test audio.wav --reference clean.wav --output results.json
  %(prog)s --demo
        """
    )
    
    parser.add_argument('--test', 
                       help='Path to test audio file', 
                       metavar='FILE')
    
    parser.add_argument('--reference', 
                       help='Path to reference audio file (for comparison)', 
                       metavar='FILE')
    
    parser.add_argument('--output', 
                       help='Output JSON file path (overrides other output flags)', 
                       metavar='FILE')
    
    parser.add_argument('--output-single', 
                       help='Output file for single audio analysis (default: single_audio_metrics.json)', 
                       metavar='FILE')
    
    parser.add_argument('--output-compare', 
                       help='Output file for comparison analysis (default: comparison_metrics.json)', 
                       metavar='FILE')
    
    parser.add_argument('--demo', 
                       action='store_true',
                       help='Run demo with generated sample audio')
    
    parser.add_argument('--examples', 
                       action='store_true',
                       help='Run hardcoded examples')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    if args.demo:
        demo_with_sample_audio()
    elif args.examples:
        example_usage()
    elif args.test:
        run_cli_analysis(args)
    else:
        print("Error: --test argument is required for analysis")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    print("Audio Quality Metrics - Example Usage")
    print("====================================")
    
    main()
