import copy
import os
import random
import threading
import time

import cv2
from playsound import playsound

import face_detect
from challenges.clf_emotion import EmotionDetect
from challenges.guess_age import GuessAge
from chatbot.blenderbot import BlenderBot
from constants import Chatbot
from face_clf import FaceRecognitionModel
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

        # chatbot = AeonaBot()
        chatbot = BlenderBot()

        face_recognition = FaceRecognitionModel()

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
                frame_predict_name = copy.deepcopy(frame)
                bbox, landmark = face_detect.scrfd_detect(frame_predict_name)
                if len(bbox) > 0:
                    user_name = face_recognition.predict(frame_predict_name, landmark)
                    if len(user_name) == 0:
                        response = f"Hello, What is your name"
                        self.speak_duration = tts(response)
                        while True:
                            guess = recognize_speech_from_mic()
                            # temp_response['transcription'] = "David"
                            # guess = temp_response

                            if guess['error'] is None:
                                message = guess["transcription"]
                                print("User: " + message)

                                user_name = face_recognition.add_user(message, frame_predict_name)
                                self.speak_duration = tts(
                                    f"Nice to meet you {user_name[:user_name.find('_')]}, How are you")
                                break
                            else:
                                self.speak_duration = tts("Can you say again?")
                    else:
                        self.speak_duration = tts(f"Nice to meet you {user_name[:user_name.find('_')]}, How are you")

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
                                        :
                                    age = int(guess_age(frame))
                                    response = f"Are you {age} years old?"
                                    self.speak_duration = tts(response)
                                    random_choice_response = ""
                                else:
                                    random_challenges = random.randint(1, 8)
                                    if random_challenges in [2, 5]:
                                        bbox, _ = face_detect.scrfd_detect(frame)
                                        bbox = bbox[0].astype(int)
                                        face_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                        if random_challenges == 2:
                                            # age
                                            response = random.choice(Chatbot.age_messages)
                                            random_choice_response = response
                                            guess_age(face_image)
                                            self.speak_duration = tts(response)

                                        else:
                                            # emotion
                                            user_emotion = emotion_detect(face_image)
                                            response = Chatbot.emotion_messages[user_emotion]
                                            self.speak_duration = tts(response)
                                    else:
                                        response = chatbot.send(message)
                                        self.speak_duration = tts(response)
                                    temp_response['transcription'] = response
                            else:
                                bbox, _ = face_detect.scrfd_detect(frame)
                                if len(bbox) > 0:
                                    self.speak_duration = tts("I can not hear you")
                                else:
                                    self.speak_duration = tts("Good bye")
                                    break
            # thread_animate.join()


if __name__ == '__main__':
    VoiceBot().run_voicebot()
