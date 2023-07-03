import numpy as np
import soundfile as sf
import librosa

def spectral_subtraction(audio, reduction_factor=1.0):
    # Compute the power spectrum of the audio signal
    audio_spec = np.abs(librosa.stft(audio))

    # Estimate the noise spectrum from a portion of the audio
    noise_frames = audio_spec[:, 188:]  # Adjust the number of frames as needed
    avg_noise_spec = np.mean(noise_frames, axis=1, keepdims=True)

    # Compute the noise reduction factor
    reduction_factor = np.maximum(0, reduction_factor)

    # Apply spectral subtraction
    enhanced_spec = audio_spec - (reduction_factor * avg_noise_spec)

    # Inverse STFT to obtain the enhanced audio signal
    enhanced_audio = librosa.istft(enhanced_spec)

    return enhanced_audio

# Example usage
# Load the audio signal
audio, sr = librosa.load('/media/ahmed/DATA/forth year/Graduation Project/tacron2/tts/6.wav', sr=None)

# Apply spectral subtraction with a reduction factor of 0.5
reduced_noise_audio = spectral_subtraction(audio, reduction_factor=0.7)

# Save the enhanced audio signal to a file
sf.write('/media/ahmed/DATA/forth year/Graduation Project/tacron2/tts/ot.wav', reduced_noise_audio, sr)
