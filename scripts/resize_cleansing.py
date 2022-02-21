import os
from PIL import Image
import cv2
from collections import Counter
import pickle

root = './105_classes_pins_dataset/'
resized_root = './pins_resized/'
cleansed_root = './pins_cleansed/'

def resizer():
    for dir in os.listdir(root):
        name_dir = root + dir
        img_list = os.listdir(name_dir)

        for idx, i in enumerate(img_list):
            img = Image.open(os.path.join(name_dir, i))
            img_resize = img.resize((256, 256), Image.ANTIALIAS)
            resized_name_dir = resized_root + dir
            try:
                img_resize.save(os.path.join(resized_name_dir, '{}.png'.format(i[:-4])), quality = 95)
            except:
                os.mkdir(resized_name_dir)
                img_resize.save(os.path.join(resized_name_dir, '{}.png'.format(i[:-4])), quality = 95)

def get_tops():
    counter = []
    try:
        with open('scores.pickle', 'rb') as pck:
            scores = pickle.load(pck)
    except:
        d = {}
        for dir in [f for f in os.listdir(resized_root) if not f.startswith(".")]:
            name_dir = resized_root + dir
            img_list = [f for f in os.listdir(name_dir) if not f.startswith(".")]
            d[dir] = []

            for idx, i in enumerate(img_list):
                img = cv2.imread(os.path.join(name_dir, i))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                fm = cv2.Laplacian(gray,cv2.CV_64F).var()
                thres = 200
                if fm >= thres:
                    d[dir].append(i)
                    counter.append(dir)
            # tops.append((dir, len(d[dir])))

        tops = Counter(counter).most_common()
        scores = [d, tops]
        with open('scores.pickle', 'wb') as pck:
            pickle.dump(scores, pck, protocol=pickle.HIGHEST_PROTOCOL)

    # tops = Counter(counter).most_common()

    return scores[0], scores[1]

def cleansing():
    d, tops = get_tops()
    names = []
    # for (a, b) in tops:
    #     names.selected.append(a)

    for i, (k, vs) in enumerate(d.items()):
        # if k in names:
        cleansed_name_dir = cleansed_root + k
        for v in vs:
            img = Image.open(os.path.join(resized_root, k, v))
            try:
                img.save(os.path.join(cleansed_name_dir, v), quality = 95)
                print("{} successfully saved".format(v))
            except:
                os.mkdir(cleansed_name_dir)
                img.save(os.path.join(cleansed_name_dir, v),  quality = 95)
                print("{} successfully saved".format(v))
        # else:
        #     continue

# s

