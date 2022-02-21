import face_recognition
from sklearn import svm
import os
import pickle
import numpy as np
# from PIL import Image

def predict(image, model_path="svm_model.clf", thres=0.2):
    with open(model_path, 'rb') as f:
            svm_clf = pickle.load(f)
    # image = face_recognition.load_image_file(img_path)
    # image = image * 255.0
    # image = image.permute(1,2,0)
    # print(image.shape, type(image), image)
    # imsave(image, "./temp_test/test01.png")
    # Image.fromarray(image).save("./temp_test/test01.png")
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return []

    image_encs = face_recognition.face_encodings(image)

    # name = svm_clf.predict(image_encs)
    probs = svm_clf.predict_proba(image_encs)

    # if image_file[:-9] == "unknown":
    #     if max(probs[0]) <= thres:
    #         name[0] = "unknown"
    # print(probs, probs.shape, type(probs))
    return probs[0]
