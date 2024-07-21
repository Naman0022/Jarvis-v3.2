# Audio Input

from errors import listenAgain,abortError
from basic import start_session

# Import Libraries
import numpy as np
import pyaudio

# Audio Input Parameters
sample_rate = 44100  # Sample rate in Hz
channels = 1  # Number of audio channels (1 for mono, 2 for stereo)
silence_threshold_multiplier = 1.5  # Multiplier for silence threshold
silence_duration = 2  # Duration of silence to stop recording (in seconds)
calibration_duration = 3  # Duration for noise calibration (in seconds)
chunk_size = 1024  # Number of frames per buffer
file_name = 'processes\\speech.wav'  # Output file name
threshold = 400
input_device = 1
p=None
mic = True

def checkNoise(re,p):
    global start_session,quit_s
    threshold = calibrate_noise(p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size,input_device_index=input_device), sample_rate, channels, calibration_duration, chunk_size)
    if threshold > 1500: print("It sure is noisy out there. Please speak loud and Clearly.")
    elif threshold > 3000:
        print("Sorry too much noise out there. I may not work properly.")
        if(input_device != 1):
            print("Please try again using a bluetooth audio device.")
            input("")
            start_session = True
        else:
            if re<5:
                return checkNoise(re+1,p)
            print("Please try again later...")
            quit_s = True
        raise listenAgain()
    return threshold

def calibrate_noise(stream, sample_rate,channels, duration, chunk_size):
    noise = []
    for _ in range(0,2):
        noise_data = []
        num_chunks = int(sample_rate * duration / chunk_size)
        for _ in range(num_chunks):
            data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
            noise_data.append(data)
        noise_data = np.concatenate(noise_data)
        noise_level = np.max(np.abs(noise_data))
        noise.append(noise_level)
    threshold = ((sum(noise) / len(noise) if len(noise) > 0 else MemoryError("noise array is of 0 length")) * silence_threshold_multiplier)  # Set threshold to a multiplier of the noise level
    if threshold < 500:
        return threshold + 100
    elif threshold > 1500:
        return threshold - 200
    return threshold

def is_silent(data, threshold):
    # Check if the maximum amplitude in the data is below the threshold
    return np.max(np.abs(data)) < threshold

def listen():
    global start_session,input_device,threshold,sample_rate
    
    import math,wave,playsound
    input_device = 2

    try:
        pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size,input_device_index=input_device)
    except OSError:
        input_device = 1
    p = pyaudio.PyAudio()
    
    if(start_session):
        start_session = False
        # Calibrate noise level
        threshold = checkNoise(1,p)
        print("Using:",p.get_device_info_by_host_api_device_index(0, input_device).get('name'))
        print(f'Noise level: {math.floor(threshold/60)}%')

    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size,input_device_index=input_device)
    playsound.playsound("audio_effects\\start.wav")

    recording = []
    silence_start = None
    silence_duration_frames = int(silence_duration * sample_rate / chunk_size)

    try:
        while True:
            data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
            recording.append(data)
            if is_silent(data, threshold):
                if silence_start is None:
                    silence_start = len(recording)
                elif len(recording) - silence_start >= silence_duration_frames:
                    break
            else:
                silence_start = None
    except KeyboardInterrupt:
        raise abortError("keyboard Interrupt - listen")
    playsound.playsound("audio_effects\\stop.wav")
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert list to numpy array
    recording = np.concatenate(recording)

    # Save the recording to a WAV file
    with wave.open(file_name, 'w') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    
    return file_name
