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
        vid = cv2.VideoCapture(0)
        tts = TextToSpeech()

        face_recognition = FaceRecognitionModel()

        emotion_detect = EmotionDetect()
        guess_age = GuessAge()

        thread_animate = threading.Thread(target=self.run_animate)
        thread_animate.start()

        while True:
            ret, frame = vid.read()

            frame_predict_name = copy.deepcopy(frame)
            bbox, landmark = face_detect.scrfd_detect(frame_predict_name)
            if len(bbox) > 0:
                is_have_human = [True, time.time()]
                user_name = face_recognition.predict(frame_predict_name, landmark)
                if len(user_name) > 0:
                    self.speak_duration = tts(
                        f"Bạn {user_name[:user_name.find('_')]} này, tránh ra cho người khác chơi nào")

                if len(user_name) == 0 and is_have_human[0]:
                    response = f"Chào bạn xinh đẹp, bạn tên là gì vậy?"
                    self.speak_duration = tts(response)
                    while True:
                        guess = recognize_speech_from_mic()

                        if guess['error'] is None:
                            message = guess["transcription"]
                            print("User: " + message)

                            user_name = face_recognition.add_user(message, frame_predict_name)
                            if len(user_name) == 0:
                                self.speak_duration = tts("Bạn nói gì, nói lại xem")
                                continue

                            self.speak_duration = tts(
                                f"Chào bạn {user_name[:user_name.find('_')]}, tôi nhớ bạn rồi đấy!")

                            time.sleep(self.speak_duration + 1)
                            self.speak_duration = tts("Giờ tôi sẽ đoán tuổi bạn, tút tút tút")

                            bbox, _ = face_detect.scrfd_detect(frame)
                            bbox = bbox[0].astype(int)
                            face_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                            # age
                            response = random.choice(Chatbot.age_messages)
                            age = guess_age(face_image)
                            self.speak_duration = tts(response.format(int(age)))

                            # emotion
                            user_emotion = emotion_detect(face_image)
                            response = Chatbot.emotion_messages[user_emotion]
                            self.speak_duration = tts(response)

                            time.sleep(self.speak_duration + 1)
                            self.speak_duration = tts("Good bye")
                            break

                        else:
                            ret, frame = vid.read()
                            bbox, _ = face_detect.scrfd_detect(frame)
                            if len(bbox) > 0:
                                is_have_human = [True, time.time()]
                                if int(time.time()) % 3:
                                    self.speak_duration = tts("Bạn nói gì, nói lại xem")
                            else:
                                if time.time() - is_have_human[1] > 10:
                                    self.speak_duration = tts("Mọi người đâu hết rồi")
                                    break


if __name__ == '__main__':
    VoiceBot().run_voicebot()
