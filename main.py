import os
import random
import threading
import time

import cv2
from playsound import playsound

import face_detect
from challenges.clf_emotion import EmotionDetect
from challenges.guess_age import GuessAge
from chatbot.aeona import AeonaBot
from constants import Chatbot
from speech.speech_to_text import recognize_speech_from_mic
from speech.text_to_speech import TextToSpeech

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VoiceBot(object):
    def __init__(self):
        self.speak_duration = 0

    def run_animate(self):
        while True:
            if self.speak_duration == 0:
                print(self.speak_duration)
                vid = cv2.VideoCapture("resources/animate_idle.mp4")
                while self.speak_duration == 0:
                    ret, frame = vid.read()
                    if ret:
                        cv2.imshow("zxc", frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    else:
                        vid = cv2.VideoCapture("resources/animate_idle.mp4")
                vid.release()

            if self.speak_duration > 0:
                print(self.speak_duration)
                vid = cv2.VideoCapture("resources/animate_speak.mp4")
                t1 = threading.Thread(target=playsound, args=("welcome.mp3",))
                t1.start()
                now = time.time()
                while time.time() - now < self.speak_duration:
                    ret, frame = vid.read()
                    if ret:
                        cv2.imshow("zxc", frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    else:
                        vid = cv2.VideoCapture("resources/animate_speak.mp4")

                self.speak_duration = 0
                t1.join()
                vid.release()

    def run_voicebot(self):
        vid = cv2.VideoCapture("/home/khanhpluto/Videos/vuong_mask.avi")
        tts = TextToSpeech()
        chatbot = AeonaBot()

        emotion_detect = EmotionDetect()
        guess_age = GuessAge()

        random_challenges = -1
        random_choice_response = ""
        temp_response = {
            "error": None,
            "transcription": "Hello"
        }

        thread_animate = threading.Thread(target=self.run_animate)
        thread_animate.start()
        while True:
            # ret, frame = vid.read()
            frame = cv2.imread("/home/khanhpluto/Downloads/download.jpeg")
            ret = True
            if ret:
                face_image, _ = face_detect.scrfd_detect(frame)
                if len(face_image) > 0:
                    message = ""
                    while True:
                        if self.speak_duration == 0:
                            guess = recognize_speech_from_mic()
                            # guess = temp_response

                            if guess['error'] is None:
                                message = guess["transcription"]
                                print("User: " + message)

                                if (len(random_choice_response) > 0 and any(
                                        x in message for x in ["yes", "yeah", "ok"])) \
                                        or all(x in message for x in ["guess", "age"]):
                                    age = int(guess_age(frame))
                                    response = f"Are you {age} years old?"
                                    self.speak_duration = tts(response)
                                else:
                                    random_challenges = random.randint(1, 8)
                                    if random_challenges == 2:
                                        # age
                                        response = random.choice(Chatbot.age_messages)
                                        self.speak_duration = tts(response)

                                    elif random_challenges == 5:
                                        # emotion
                                        # ret, frame = vid.read()
                                        frame = cv2.imread("/home/khanhpluto/Downloads/download.jpeg")
                                        user_emotion = emotion_detect(frame)
                                        response = Chatbot.emotion_messages[user_emotion]
                                        self.speak_duration = tts(response)
                                    else:
                                        response = chatbot.send(message)
                                        self.speak_duration = tts(response)
                                    temp_response['transcription'] = response
                            else:
                                face_image, _ = face_detect.scrfd_detect(frame)
                                if len(face_image) > 0:
                                    self.speak_duration = tts("I can not hear you")
                                else:
                                    self.speak_duration = tts("Good bye")
                                    break
            thread_animate.join()


if __name__ == '__main__':
    VoiceBot().run_voicebot()
