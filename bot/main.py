from basic import json,history,setup,engine
from processor import process,chatAI,reply
from audio import listen
from preprocessor import preprocessing
from errors import Quiterror,listenAgain


def main():
    global mic,txt,setup
    mic = True
    while True:

        try:
            audio_path = listen()
            a_txt = preprocessing(audio_path,1)
            process(a_txt)

        except Quiterror as q:
            txt = "Now stop your execution,bye"
            chatAI(txt)
            engine.stop()
            setup['history'].append(history)
            with open("processes\\setup.json", 'w') as json_file:
                json.dump(setup, json_file, indent=4)
            print("_"*10,"END","_"*10)
            return

        except listenAgain as v:
            pass

        except KeyboardInterrupt:
            print("Interrupted by ctrl+c")
            return

if __name__ == "__main__":
    print("_"*10,"START","_"*10)
    print("Preparing please wait...")
    start_session = True
    main()