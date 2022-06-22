from constants import Download
from face_detect.scrfd import SCRFD

_detector = SCRFD(
    model_file=Download.download_url["scrfd_face_detect"][0])
_detector.prepare(-1)


def scrfd_detect(img):
    bboxes, kps = _detector.detect(img, 0.4, input_size=(640, 640))

    w, h, _ = img.shape
    for i in range(len(bboxes)):
        box = bboxes[i]
        if box[2] - box[0] > w / 3 and box[3] - box[1] > h / 3:
            return bboxes[i:i + 1], kps[i:i + 1]
    return [], []
