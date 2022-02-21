import face_recognition
from PIL import Image
import os
from tqdm import tqdm

root = '../datasets/lfw/'
rroot = '../datasets/lfw_resized/'

for dir in os.listdir(root):
    # img_list = os.listdir(root + dir):
    for i, img in enumerate(tqdm([f for f in os.listdir(root + dir) if not f.startswith(".")])):
        try:
            path = os.path.join(root, dir, img)
            imgLoaded = face_recognition.load_image_file(path)
            # print(path)
            top, right, bottom, left = face_recognition.face_locations(imgLoaded)[0]
            intv1 = bottom-top
            intv2 = right-left
            face = Image.fromarray(imgLoaded[int(top-0.05*intv1):int(top-0.05*intv1)+int(intv1*1.1), int(left-0.05*intv2):int(left-0.05*intv2)+int(intv2*1.1)])
            face_resized = face.resize((256, 256), Image.ANTIALIAS)
            if not os.path.exists(rroot + dir):
                os.mkdir(rroot + dir)
            face_resized.save(os.path.join(rroot, dir, img))
        except:
            continue