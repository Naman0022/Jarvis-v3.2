import pyttsx3,playsound,time,json

# Basic Functions
def load_JSON(path):

    with open(path, 'r') as file:
        data = json.load(file)
    return data

# Basic globals
start_session=True
setup = load_JSON("processes\\setup.json")
background = False
history = []
engine = pyttsx3.init()


try:
    import spacy
    nlp = spacy.load("en_core_web_md")
    lemmatizer = nlp.get_pipe("lemmatizer")
except Exception as e:
    print("Error with scapy")

def express(_express):
    playsound.playsound("audio_effects/"+str(_express)+".wav")
    time.sleep(0.1)

def print_speak(txt):
    voices = engine.getProperty('voices')
    engine.setProperty('voice',voices[1].id)
    engine.setProperty('rate', 180)
    parts = txt.split("*")
    expression = ["sigh","sighs","wink","winks"] #,"facepalm","eyeroll","eyerolls"
    for part in parts:
        if part == "":
            continue
        elif part in [".","?","!"]:
            print(part,end="")
        elif nlp(part.lower())[0].lemma_ in expression:
            print("*"+nlp(part)[0].lemma_+"* ",end="")
            express(nlp(part)[0].lemma_)
        else:
            print(part,end="")
            engine.runAndWait()
            engine.say(part) 
    
    engine.runAndWait()
    return
