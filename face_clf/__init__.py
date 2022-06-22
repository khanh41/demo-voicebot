import uuid

import cv2
from sklearn import neighbors

from face_detect import scrfd_detect
from face_embedding import arcface_inference


def get_name(message: str):
    if "tên" in message:
        if "là" in message:
            is_index = message.find("là") + 3
        else:
            is_index = message.find("tên") + 4

        end_index = message.find(" ", is_index)
        if end_index == -1:
            return message[is_index:]
        return message[is_index:end_index]

    if " " not in message:
        return message
    return ""


class FaceRecognitionModel:
    def __init__(self, company="BAP"):
        self.X = []
        self.X_mask = []
        self.y = []
        self.company = company
        self.classifier = None
        self.classifier_mask = None

    def train_with_data_loaded(self):
        self.classifier.fit(self.X, self.y)
        self.classifier_mask.fit(self.X_mask, self.y)

    def predict(self, X_frame, landmarks_coord=None):
        if self.classifier is None:
            return ""

        faces_encodings = arcface_inference(X_frame, landmarks_coord)

        # Use the KNN model to find the best matches for the test face
        closest_distances = self.classifier.kneighbors(faces_encodings, n_neighbors=1)
        closest_distances_mask = self.classifier_mask.kneighbors(faces_encodings, n_neighbors=1)

        if closest_distances[0][0] <= 1.1:
            pred = self.classifier.predict(faces_encodings)[0]
        elif closest_distances_mask[0][0] <= 1.1 - 0.1:
            pred = self.classifier_mask.predict(faces_encodings)[0]
        else:
            pred = ""

        return pred

    def add_user(self, message, image):
        user_name = get_name(message)
        if len(user_name) == 0:
            return user_name

        user_name = f"{user_name}_{uuid.uuid4()}"

        if user_name in self.y:
            raise Exception(f"{user_name} already exist")

        face_locations, landmarks = scrfd_detect(image)
        self.X.append(arcface_inference(image, landmarks)[0])

        image[int(landmarks[0][2][1]) - 15:, :, :] *= 0
        self.X_mask.append(arcface_inference(image, landmarks)[0])

        self.y.append(user_name)

        if self.classifier is None:
            self.classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',
                                                             weights='distance')
            self.classifier_mask = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',
                                                                  weights='distance')
        self.train_with_data_loaded()

        cv2.imwrite(f'resources/{user_name}.png', image)

        return user_name
