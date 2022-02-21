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

dataset_path = "../datasets/pins_celebs"

def ext_single(path):

    img = cv2.imread(path)
    # img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
    # if img.shape[0] != img.shape[1]:
       
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    faces = detector(gray, 1)
    landmarks = []
    for i, face in enumerate(faces):
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
    # cv2.imwrite('./test_{}'.format())
    cv2.imwrite('./test2/images/test.png', imgMasked)
    cv2.dilate(res, (1,1))
    cv2.imwrite('./test2/masks/test.png', res)
    cv2.imwrite('./test2/orijins/test.jpg', img)

def ext():
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    img_root = "../datasets/pins_cleansed/"
    # count = 0
    for dir in os.listdir(img_root):
        name_root = img_root + dir
        count = 0
        for i, img in enumerate(os.listdir(name_root)):
            count += 1
            idx = i
            # print(img)
            img_path = os.path.join(name_root, img)
            # print(img_path)


            img = cv2.imread(img_path)
            # img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
            # if img.shape[0] != img.shape[1]:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                continue

            detector = dlib.get_frontal_face_detector()

            predictor_path = "shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(predictor_path)

            faces = detector(gray, 1)
            landmarks = []
            for i, face in enumerate(faces):
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
            try:
                imgMasked = cv2.fillPoly(imgMasked, np.array([landmarks]), (255,255,255))
            except:
                continue

            res = np.zeros(img.shape)

            res = cv2.fillPoly(res, np.array([landmarks]), (255,255,255))
            # cv2.imwrite('./test_{}'.format())
            print(img_path)
            cv2.imwrite('../datasets/pins_celebs/images/{}_{:04d}.png'.format(dir[5:], count), imgMasked)
            cv2.dilate(res, (1,1))
            cv2.imwrite('../datasets/pins_celebs/masks/{}_{:04d}.png'.format(dir[5:], count), res)
            cv2.imwrite('../datasets/pins_celebs/originals/{}_{:04d}.png'.format(dir[5:], count), img)
            

# if __name__ == "main":
#     ext_single()

# def move_origin():
#     img_root = "../datasets/pins_cleansed/"
#     move_to = "../datasets/pins_celebs/originals"
#     # count = 0
#     for dir in os.listdir(img_root):
#         name_root = img_root + dir
#         # print(name_root)
#         count = 0
#         for i, img in enumerate(os.listdir(name_root)):
#             print(img)
#             count += 1
#             os.rename(os.path.join(name_root, img), os.path.join(move_to, "{}_{:04d}.png".format(dir[5:], count)))

def process():
    deprecated = []
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    img_root = "../datasets/pins_cleansed/"
    # count = 0
    for dir in os.listdir(img_root):
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
        print(end - start)



def extr_mp(data):
    count = data[0]
    img_path = data[1]
    name = img_path.split("/")[-2][5:]
    try:
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        # if img.shape[0] != img.shape[1]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()

        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)

        faces = detector(gray, 1)
        landmarks = []
        for i, face in enumerate(faces):
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
        print(img_path)
        cv2.imwrite('../datasets/pins_celebs/images/{}_{:04d}.png'.format(name, count), imgMasked)
        cv2.dilate(res, (1,1))
        cv2.imwrite('../datasets/pins_celebs/masks/{}_{:04d}.png'.format(name, count), res)
        # cv2.imwrite('../datasets/pins_celebs/originals/{}_{:04d}.png'.format(dir[5:], count), img)
        shutil.copy(img_path, os.path.join("../datasets/pins_celebs/originals", "{}_{:04d}.png".format(name, count)))
    except:
        with open("deprecated_files.txt", "a") as f:
            f.write(img_path)
        print("{} is deprecated".format(img_path))