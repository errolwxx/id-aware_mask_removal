import os
import cv2

images_path = "../datasets/pins_celebs/originals"
masks_path = "../datasets/pins_celebs/masks_backup"
save_path = "../datasets/pins_celebs/masks"

def rename_images():
    for f in os.listdir(images_path):
        filename = os.path.join(images_path, f)
        if len(filename.split(" ")) == 1:
            continue
        else:
            os.rename(filename, "_".join(filename.split(" ")))

def rename_masks():
    for f in os.listdir(masks_path):
        filename = os.path.join(masks_path, f)
        if len(filename.split(" ")) == 1:
            continue
        else:
            os.rename(filename, "_".join(filename.split(" ")))

def to_gray():
    for f in os.listdir(masks_path):
        print(f)
        filename = os.path.join(masks_path, f)
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(filename, gray.shape)
        savename = os.path.join(save_path, f)
        cv2.imwrite(savename, gray)