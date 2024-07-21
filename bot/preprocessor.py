# Text Processing
from basic import print_speak
from errors import listenAgain

import whisper

model = whisper.load_model("base")

def preprocessing(path,calls):
    result = model.transcribe(path, fp16=False)

    if "error" in result or "text" not in result:
        if calls < 6:
            return preprocessing(path,calls+1)
        else:
            print_speak("please say that again...")
            raise listenAgain()
        
    return result["text"]
    # print(result["text"])
    # API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
    # headers = {"Authorization": "Bearer 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"}

    # def query(filename):
    #     with open(filename, "rb") as f:
    #         data = f.read()
    #     response = requests.post(API_URL, headers=headers, data=data)
    #     return response.json()

    # results = query(path)

    # if "error" in results or "text" not in results:
    #     if calls < 6:
    #         return preprocessing(path,calls+1)
    #     else:
    #         print_speak("please say that again...")
    #         raise listenAgain()
        
    # if (results["text"] == "..." or results["text"] == " " or results["text"] == ""):
    #     print_speak("please say that again...")
    #     raise listenAgain()
    
    # return tokenize(results["text"])