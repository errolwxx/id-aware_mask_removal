# import dlib
# import cv2

# detector = dlib.get_frontal_face_detector()

# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# img = cv2.imread("0.jpg")

# win = dlib.image_window()
# win.set_image(img)

# faces = detector(img, 1)

# for i, d in enumerate(faces):
#     print(i+1, "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())
#     shape = predictor(img, faces[i])
#     win.add_overlay(shape)

# win.add_overlay(faces)
# dlib.hit_enter_to_continue()

import cv2
import dlib
import numpy as np
import os
import time
import shutil
from multiprocessing import Pool
from tqdm import tqdm

dataset_path = "../datasets/lfw_mask"


def process():
    deprecated = []
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    img_root = "../datasets/lfw_resized/"
    # count = 0
    for dir in tqdm(os.listdir(img_root)):
        start = time.time()
        name_root = img_root + dir
        temp = os.listdir(name_root)
        # count = 0
        pool = Pool(8)
        data = zip(list(range(1, len(temp))), [os.path.join(name_root, f) for f in temp])
        pool.map(extr_mp, data)
        pool.close()
        pool.join()
        end = time.time()
        # print(end - start)



def extr_mp(data):
    count = data[0]
    img_path = data[1]
    name = img_path.split("/")[-2]
    try:
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        # if img.shape[0] != img.shape[1]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()

        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)

        faces = detector(gray, 1)
        face = faces[0]
        landmarks = []
        shape = predictor(img, face)
        shapeLst = shape.parts()
        for j, pt in enumerate(shapeLst):
            # print(pt)
            # pt_pos = (pt.x, pt.y)
            if j <= 29:
                if (j == 1)|(j==14):
                # pt_pos = [(pt.x+shapeLst[j+1].x)/2, (pt.y+shapeLst[j+1].y)/2]
                    pt_pos = [int((pt.x+shapeLst[j+1].x)/2), int((pt.y+shapeLst[j+1].y)/2)]
                    # print(pt_pos)
                    landmarks.append(list(pt_pos))
                elif (j in range(3, 14)) | (j == 29):
                    pt_pos = (pt.x, pt.y)
                    # print(pt_pos)
                    landmarks.append(list(pt_pos))
                else:
                    continue
            else: 
                break
                # landmarks.append(list(pt_pos))
                # cv2.circle(img, pt_pos, 1, (255,0, 0), 2)
                # cv2.putText(img, str(j+1), pt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        imgMasked = img.copy()
        # imgMasked = cv2.drawContours(imgMasked, np.array([landmarks]), 0, (0, 255, 0), 3)
        imgMasked = cv2.fillPoly(imgMasked, np.array([landmarks]), (255,255,255))


        res = np.zeros(img.shape)

        res = cv2.fillPoly(res, np.array([landmarks]), (255,255,255))
        # print([landmarks])
        # print(img_path)
        cv2.imwrite('../datasets/lfw_mask/images/{}_{:04d}.png'.format(name, count), imgMasked)
        cv2.dilate(res, (1,1))
        cv2.imwrite('../datasets/lfw_mask/masks/{}_{:04d}.png'.format(name, count), res)
        # cv2.imwrite('../datasets/pins_celebs/originals/{}_{:04d}.png'.format(dir[5:], count), img)
        # shutil.copy(img_path, os.path.join("../datasets/lfw_mask/originals", "{}_{:04d}.png".format(name, count)))
    except:
        with open("deprecated_files_lfw.txt", "a") as f:
            f.write(img_path)
        print("{} is deprecated".format(img_path))