import subprocess
import threading
import time
from typing import Any

from gtts import gTTS
from playsound import playsound


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


class TextToSpeech:
    language = 'en'

    def __init__(self) -> None:
        pass

    def __call__(self, mytext, *args: Any, **kwds: Any) -> Any:
        print("Bot: " + mytext)
        myobj = gTTS(text=mytext, lang=self.language, slow=False)

        mp3_name = "welcome.mp3"
        myobj.save(mp3_name)
        # time.sleep(1)
        return get_length(mp3_name)
