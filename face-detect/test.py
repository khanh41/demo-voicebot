import cv2
import numpy as np

from base_model.retinaface import RetinaModel

img = cv2.imread('/home/khanhpluto/Pictures/datatest/namtp/Screenshot from 2021-12-20 11-38-01.png')
print(img.shape)
im_shape = img.shape

im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
# im_scale = 1.0
# if im_size_min>target_size or im_size_max>max_size:
im_scale = float(RetinaModel.target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > RetinaModel.max_size:
    im_scale = float(RetinaModel.max_size) / float(im_size_max)

print('im_scale', im_scale)

scales = [im_scale]
flip = False

faces = None
landmarks = None
import time

now = time.time()
for c in range(RetinaModel.count):
    faces, landmarks = RetinaModel.detector.detect(img,
                                                   RetinaModel.thresh,
                                                   scales=scales,
                                                   do_flip=flip)
    print(c, faces.shape, landmarks.shape)

print(time.time() - now)
if faces is not None:
    print("zxc")