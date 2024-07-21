from basic import print_speak

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

def volume(x):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Set Master Volume level
    volume.SetMasterVolumeLevelScalar(x / 100.0, None)
    print_speak("Done")
    print("returning volume")
    return "volume"