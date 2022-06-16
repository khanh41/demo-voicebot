import random

import cv2

import face_detect
from challenges.clf_emotion import EmotionDetect
from challenges.guess_age import GuessAge
from chatbot.aeona import AeonaBot
from constants import Chatbot
from speech.speech_to_text import recognize_speech_from_mic
from speech.text_to_speech import TextToSpeech

if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    tts = TextToSpeech()
    chatbot = AeonaBot()

    emotion_detect = EmotionDetect()
    guess_age = GuessAge()

    random_challenges = -1
    random_choice_response = ""
    print("Start")
    while True:
        ret, frame = vid.read()
        if ret:
            face_image, _ = face_detect.scrfd_detect(frame)
            if len(face_image) > 0:
                tts("Hello")
                message = ""
                while True:
                    print("Say something")
                    guess = recognize_speech_from_mic()

                    if guess['error'] is None:
                        message = guess["transcription"]
                        print(f'User: {message}')

                        if len(random_choice_response) > 0 and any(x in message for x in ["yes", "yeah", "ok", "okay"]):
                            age = guess_age(frame)
                            tts(f"Are you {age} years old?")  # number to text
                        else:
                            random_challenges = random.randint(1, 8)
                            if random_challenges == 2:
                                # age
                                random_choice_response = random.choice(Chatbot.age_messages)
                                tts(random_choice_response)

                            elif random_challenges == 5:
                                # emotion
                                ret, frame = vid.read()
                                user_emotion = emotion_detect(frame)
                                tts(Chatbot.emotion_messages[user_emotion])
                            else:
                                response = chatbot.send(message)
                                if len(response) > 3:
                                    print(f'Bot: {response}')
                                    tts(response)

                    else:
                        face_image, _ = face_detect.scrfd_detect(frame)
                        if len(face_image) > 0:
                            tts("I can not hear you")
                        else:
                            tts("Good bye")
                            break
