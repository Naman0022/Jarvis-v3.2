class Quiterror(RuntimeError):
    def __init__(self, txt):
        self.value = txt
    def __str__(self):
        return(self.value)

class listenAgain(RuntimeError):
    def __init__(self):
        return
    
class abortError(RuntimeError):
    def __init__(self, txt):
        self.value = txt
    def __str__(self):
        return(self.value)