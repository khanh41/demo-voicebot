from constants import Download
from face_detect.scrfd import SCRFD

_detector = SCRFD(
    model_file=Download.download_url["scrfd_face_detect"][0])
_detector.prepare(-1)


def scrfd_detect(img):
    bboxes, kps = _detector.detect(img, 0.4, input_size=(640, 640))
    return bboxes, kps
