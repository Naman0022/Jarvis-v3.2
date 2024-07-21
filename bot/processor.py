import concurrent.futures

from basic import setup, print_speak, nlp
from errors import Quiterror,listenAgain
from functions import volume


messages = []
reply=""

token,entity,propnoun,noun,verb,adjective,number,tense,reply,txt = None,None,None,None,None,None,None,None,None,None

noCommand = 0

from openai import OpenAI
import re
# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

for system_instrn in ['personality', 'chat']:
    messages.append({"role": "system", "content": setup[system_instrn]})

for history_message in setup['history'][-10:]:
    messages.append({"role": "user", "content": history_message})

def chatAI(txt):
    global reply,setup
    print("USER >>",txt)
    messages.append({"role": "user", "content": txt})
    completion = client.chat.completions.create(
        model="Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF",
        messages=messages,
        temperature=0.7,
        stream=True,
    )

    buffer = ""
    message = ""
    print("BOT >> " ,end="")
    for chunk in completion:
        if chunk.choices[0].delta.content:
            buffer += chunk.choices[0].delta.content
            # Check if buffer contains complete sentences
            sentences = re.split(r"(?<=[?.:\n!])", buffer)
            for i, sentence in enumerate(sentences):
                if i < len(sentences) - 1:  # All except the last one are complete
                    print_speak(sentence)
                    message += sentence
                else:
                    buffer = sentence  # Last one might be incomplete, save it in buffer

    if buffer:
        print_speak(buffer)
        message += buffer
    setup['history'].append(txt)


def setvolume():
    global v_level
    v_level = 50
    if "volume" in token:
        if number:
            if int(number[0]) > 100:
                return volume(int(100))
        elif "up" in token:
            return volume(v_level+10)
        elif "down" in token:
            return volume(v_level-10)

def os():
    import os
    global background
    if token:
        if 'restart' in token:
            if number:
                return os.system("shutdown /r /t " + str(number[0]))
            return os.system("shutdown /r /t 5")
        if 'shut' in token and "down" in token:
            if number:
                return os.system("shutdown /s /t " + str(number[0]))
            return os.system("shutdown /s /t " + str(5))
        if 'log' in token and "off" in token:
            background=True
            os.system("shutdown /l")
            return "logging off"

def check_condition_4():
    if True in [nlp(str(v)).similarity(nlp("generate"))>0.5 for v in token] and "image" in token:
        return "generate:Image"

def battery():
    if noun and "battery" in noun:
        return "battery:status"
    else:
        return None

def quit():
    if token and "stop" in token and ("execution" in token or "program" in token) or "quit" in token:
        raise Quiterror("I have to go now, bye!")

# def check_condition_7():
#     # Your condition here
#     if False:  # Example condition
#         return "Condition 7 is true"

# def check_condition_8():
#     # Your condition here
#     if False:  # Example condition
#         return "Condition 8 is true"

def process(_txt):
    global token,entity,propnoun,noun,verb,adjective,number,tense,reply
          
    token = [token.lemma_.lower() for token in nlp(_txt) if token.pos_ != 'PUNCT' and not token.is_stop]
    number = [token.lemma_ for token in nlp(_txt) if token.pos_ == 'NUM']
    conditions = [
        battery,
        quit,
        os,
        setvolume,
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(condition): condition for condition in conditions}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                raise listenAgain
        chatAI(_txt)
        return