import numpy as np
import soundfile as sf
import os

def get_stego_audio(input_file="static/audio/normal_lofi.wav", output_file="static/audio/therapeutic_lofi.wav"):
    if os.path.exists(output_file):
        return output_file
    
    if not os.path.exists(input_file):
        return None

    carrier_data, sample_rate = sf.read(input_file)
    if len(carrier_data.shape) > 1:
        carrier_data = carrier_data[:, 0]
        
    num_samples = len(carrier_data)
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    payload = np.sin(2 * np.pi * 432.0 * t) * 0.03
    stego_audio = carrier_data + payload
    
    # Normalize
    max_val = np.max(np.abs(stego_audio))
    if max_val > 1.0: stego_audio /= max_val
        
    sf.write(output_file, stego_audio, sample_rate)
    return output_file