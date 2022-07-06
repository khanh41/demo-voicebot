import copy
import os
import random
import threading
import time

import cv2

import face_detect
# from challenges.clf_emotion import EmotionDetectKNN
from challenges.guess_age import GuessAgeKNN
from constants import Chatbot
from face_clf import FaceRecognitionModel
from speech.text_to_speech import TextToSpeech

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VoiceBot(object):
    def __init__(self):
        self.speak_duration = 0
        self.user_frame = None
        self.user_frame_predict = None
        self.username = ""
        self.bbox = []
        self.landmark = []

    def predict_bbox(self):
        while True:
            if self.user_frame is not None:
                self.user_frame_predict = copy.deepcopy(self.user_frame)
                self.bbox, self.landmark = face_detect.scrfd_detect(self.user_frame_predict)

    def run_animate(self):
        vid_user = cv2.VideoCapture("/home/khanhpluto/Downloads/zxczxc/data_train_22102021_cam72.avi")
        while True:
            ret_user, self.user_frame = vid_user.read()
            if ret_user:
                draw_frame = copy.deepcopy(self.user_frame)
                if len(self.bbox) > 0:
                    x1, y1, x2, y2, _ = self.bbox[0].astype(int)
                    draw_frame = cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                cv2.imshow("User", draw_frame)
            else:
                vid_user = cv2.VideoCapture("/home/khanhpluto/Videos/vuong_no_mask.avi")
        vid_user.release()

    def run_voicebot(self):
        tts = TextToSpeech()

        face_recognition = FaceRecognitionModel()

        guess_age = GuessAgeKNN()

        thread_animate = threading.Thread(target=self.run_animate)
        thread_animate.start()

        thread_bbox = threading.Thread(target=self.predict_bbox)
        thread_bbox.start()

        while True:
            if len(self.bbox) > 0:
                temp_landmark = copy.deepcopy(self.landmark)
                temp_user_frame = copy.deepcopy(self.user_frame_predict)
                user_name = face_recognition.predict(temp_user_frame, temp_landmark)
                if len(user_name) > 0:
                    self.username = user_name[:user_name.find('_')]
                    self.speak_duration = tts(
                        f"Tôi đã nói là bạn {self.username} rồi mà")
                    time.sleep(self.speak_duration + 2)

                else:
                    face_recognition.add_user("temp", temp_user_frame, temp_landmark)
                    self.speak_duration = tts("Giờ tôi sẽ đoán tuổi bạn, giữ yên mặt nha, tút tút tút")

                    # age
                    response = random.choice(Chatbot.age_messages)
                    ages = []
                    usernames = []
                    temp_landmark = None
                    temp_user_frame = None
                    while len(ages) < 6:
                        temp_landmark = copy.deepcopy(self.landmark)
                        temp_user_frame = copy.deepcopy(self.user_frame_predict)
                        if len(temp_landmark) > 0:
                            user_name = face_recognition.predict(temp_user_frame, temp_landmark)
                            usernames.append(user_name)

                            if len(user_name) > 0:
                                if user_name == "temp":
                                    age = guess_age(temp_user_frame, temp_landmark)
                                    ages.append(int(age))
                                else:
                                    self.username = user_name[:user_name.find('_')]
                                    self.speak_duration = tts(
                                        f"Tôi đã nói là bạn {self.username} rồi mà")
                                    usernames = []
                                    time.sleep(self.speak_duration + 2)
                                    break

                    if usernames.count("temp") < 3:
                        face_recognition.delete_user_temp()
                        continue

                    time.sleep(self.speak_duration)
                    age = sum(ages) // len(ages)
                    self.speak_duration = tts(response.format(age))

                    face_recognition.delete_user_temp()
                    face_recognition.add_user(age, temp_user_frame, temp_landmark)

                    time.sleep(self.speak_duration + 1)
                    time.sleep(2)


if __name__ == '__main__':
    VoiceBot().run_voicebot()
