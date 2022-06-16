import os
from typing import Any

from gtts import gTTS


class TextToSpeech:
    language = 'en'

    def __init__(self) -> None:
        pass

    def __call__(self, mytext, *args: Any, **kwds: Any) -> Any:
        myobj = gTTS(text=mytext, lang=self.language, slow=False)
        myobj.save("welcome.mp3")
        os.system("welcome.mp3")
