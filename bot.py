# Basic Functions
def load_JSON(path):
    import json
    with open(path, 'r') as file:
        data = json.load(file)
    return data
# Basic globals
import time
start_session=True
quit_s = False
setup = load_JSON("processes\\setup.json")
background = False
history = []

# Text Processing
import spacy
nlp = spacy.load("en_core_web_md")
lemmatizer = nlp.get_pipe("lemmatizer")
txt,token,entity,propnoun,noun,verb,adjective,number,tense,reply = None,None,None,None,None,None,None,None,None,None

# Text-to-Speech
whisper_api_calls = 0

# lemma-Instruct Parameters
from huggingface_hub import InferenceClient
client = InferenceClient(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token="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",
    )
messages = []

count=0

for system_instrn in ['preInstruction', 'personality', 'chat']:
    messages.append({"role": "system", "content": setup[system_instrn]})

for history_message in setup['history'][-10:]:
    messages.append({"role": "system", "content": history_message})

# Audio Input

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
            return listen()
        else:
            if re<5:
                return checkNoise(re+1,p)
            print("Please try again later...")
            quit_s = True
            raise ValueError("Re")
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
    global start_session,input_device,threshold,sample_rate,mic

    import math,time,wave,playsound
    
    input_device = 1
    if mic:
        input_device = 2

    p = pyaudio.PyAudio()


    if(start_session):
        start_session = False


        # Calibrate noise level
        try:
            threshold = checkNoise(1,p)
        except OSError:
            mic = False
            start_session = True
            listen()
        print("Using:",p.get_device_info_by_host_api_device_index(0, input_device).get('name'))
        print(f'Noise level: {math.floor(threshold/60)}%')
          
    # Open audio stream
    start = time.process_time()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size,input_device_index=input_device)
    except OSError:
        mic = False
        start_session = True
        listen()
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
        pass
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
    
    # print("-",round(time.process_time() - start,2),"sec")

def preprocessing(path,calls):
    import requests

    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
    headers = {"Authorization": "Bearer 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"}

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    results = query(path)

    if "error" in results or "text" not in results:
        if calls < 6:
            return preprocessing(path,calls+1)
        else:
            print_speak("please say that again...")
            raise ValueError("Re")
        
    if (results["text"] == "..." or results["text"] == " " or results["text"] == ""):
        print_speak("please say that again...")
        raise ValueError("Re")
    
    # with open("processes/log.txt", "a") as file:
    #         file.write(("_"*20))
    #         file.write("\n"+str(datetime.datetime.now())+"\n")
    #         file.write("User >> "+str(results["text"])+"\n")
    return tokenize(results["text"])



import concurrent.futures,re,threading
from googlesearch import search


class Quiterror(RuntimeError):
    def __init__(self, txt):
        self.value = txt
    def __str__(self):
        return(self.value)

def casual():
    global reply,txt
    print("USER >>",txt)
    setup['history'].append(txt)
    messages.append({"role": "user", "content": str(txt)})
    message = client.chat_completion(
        messages=messages,
        temperature=0.9,
        max_tokens=6000,
        stream=False,
    )
    reply = message.choices[0].message.content

# def google_search(query, num_results=5):
#     try:
#         for i, result in enumerate(search(query, num_results=num_results, stop=num_results)):
#             print(f"Result {i+1}: {result}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

def check_condition_1():

    # nlp(str(v)).similarity(nlp("find"))>0.5

    if True in ["find" in verb+noun] or True in ["information" in verb+noun]:
        # if 'linkden' in token or 'linden' in token:
        #     print("case1")
        #     text = "".join([ent+" " for ent in entity if ent != "linkden" or ent != "linden"])
        #     if text == "":
        #         text = "".join([ent+" " for ent in propnoun if ent != "information"])
        #     if text == "":
        #         text = "".join([ent+" " for ent in noun if ent != "information"])  
        

        # if "google" in token:
        #     return "google"
        pass

def check_condition_2():
    global v_level
    v_level = 50
    if "volume" in noun:
        if number:
            if int(number[0]) > 100:
                print_speak("Please tell be a number between 0-100")
                return " "
            return volume(int(number[0]))
        elif "up" in token:
            return volume(v_level+10)
        elif "down" in token:
            return volume(v_level-10)

def check_condition_3():
    import os
    global background
    if 'restart' in verb:
        if number:
            return os.system("shutdown /r /t " + str(number[0]))
        return os.system("shutdown /r /t 5")
    if 'shut' in verb and "down" in adjective:
        if number:
            return os.system("shutdown /s /t " + str(number[0]))
        return os.system("shutdown /s /t " + str(5))
    if 'log' in verb and "off" in adjective:
        background=True
        os.system("shutdown /l")
        return "logging off"

def check_condition_4():
    if True in [nlp(str(v)).similarity(nlp("generate"))>0.5 for v in verb] and "image" in noun:
        return "generate:Image"

def check_condition_5():
    if "battery" in noun:
        return "battery:status"

def check_condition_6():
    global quit_s,txt
    if "stop" in token and ("execution" in token or "program" in token) or "quit" in token:
        # quit_s = True
        raise Quiterror("I have to go now, bye!")
        # txt = "I have to go now"
        # casual()
        # print_speak(reply)
        # print(quit_s,"raised error")

# def check_condition_7():
#     # Your condition here
#     if False:  # Example condition
#         return "Condition 7 is true"

# def check_condition_8():
#     # Your condition here
#     if False:  # Example condition
#         return "Condition 8 is true"

def process():
    conditions = [
        check_condition_1,
        check_condition_2,
        check_condition_3,
        check_condition_4,
        check_condition_5,
        check_condition_6,
        casual,
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(condition): condition for condition in conditions}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                # with open("processes/log.txt", "a") as file:
                #     file.write("BOT >> "+result+"\n")
                return True
        return False
    
def volume(x):
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    from ctypes import cast, POINTER

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Set Master Volume level
    volume.SetMasterVolumeLevelScalar(x / 100.0, None)
    print_speak("Done")
    print("returning volume")
    return "volume"

def session():
    global count
    print("session",count)
    count+=1
    import json
    global reply,mic,setup,quit_s,txt
    mic = True
    while True:
        if quit_s:
            setup['history'].append(history)
            print("dumppinf")
            with open("processes\\setup.json", 'w') as json_file:
                json.dump(setup, json_file, indent=4)
            print("_"*10,"END","_"*10)
            return
        try:
            audio_path = listen()
            preprocessing(audio_path,1)
            if(not process()):
                print_speak(reply)   

        except Quiterror as q:
            txt = q
            casual()
            print_speak(reply)
            return

        except ValueError as v:
            print(v)
            pass

        except KeyboardInterrupt:
            print("Interrupted by ctrl+c")
            return

print("_"*10,"START","_"*10)
print("Preparing please wait...")
start_session = True
session()