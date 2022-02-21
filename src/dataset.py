import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread, imsave
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

    #     self.labels = ['Adriana_Lima', 'Alex_Lawther', 'Alexandra_Daddario',
    #    'Alvaro_Morte', 'Amanda_Crew', 'Andy_Samberg', 'Anne_Hathaway',
    #    'Anthony_Mackie', 'Avril_Lavigne', 'Ben_Affleck', 'Bill_Gates',
    #    'Bobby_Morley', 'Brenton_Thwaites', 'Brian_J._Smith',
    #    'Brie_Larson', 'Chris_Evans', 'Chris_Hemsworth', 'Chris_Pratt',
    #    'Christian_Bale', 'Cristiano_Ronaldo', 'Danielle_Panabaker',
    #    'Dominic_Purcell', 'Dwayne_Johnson', 'Eliza_Taylor',
    #    'Elizabeth_Lail', 'Emilia_Clarke', 'Emma_Stone', 'Emma_Watson',
    #    'Gwyneth_Paltrow', 'Henry_Cavil', 'Hugh_Jackman', 'Inbar_Lavi',
    #    'Irina_Shayk', 'Jake_Mcdorman', 'Jason_Momoa', 'Jennifer_Lawrence',
    #    'Jeremy_Renner', 'Jessica_Barden', 'Jimmy_Fallon', 'Johnny_Depp',
    #    'Josh_Radnor', 'Katharine_Mcphee', 'Katherine_Langford',
    #    'Keanu_Reeves', 'Krysten_Ritter', 'Leonardo_DiCaprio',
    #    'Lili_Reinhart', 'Lindsey_Morgan', 'Lionel_Messi', 'Logan_Lerman',
    #    'Madelaine_Petsch', 'Maisie_Williams', 'Maria_Pedraza',
    #    'Marie_Avgeropoulos', 'Mark_Ruffalo', 'Mark_Zuckerberg',
    #    'Megan_Fox', 'Miley_Cyrus', 'Millie_Bobby_Brown',
    #    'Morena_Baccarin', 'Morgan_Freeman', 'Nadia_Hilker',
    #    'Natalie_Dormer', 'Natalie_Portman', 'Neil_Patrick_Harris',
    #    'Pedro_Alonso', 'Penn_Badgley', 'Rami_Malek', 'Rebecca_Ferguson',
    #    'Richard_Harmon', 'Rihanna', 'Robert_De_Niro', 'Robert_Downey_Jr',
    #    'Sarah_Wayne_Callies', 'Selena_Gomez', 'Shakira_Isabel_Mebarak',
    #    'Sophie_Turner', 'Stephen_Amell', 'Taylor_Swift', 'Tom_Cruise',
    #    'Tom_Hardy', 'Tom_Hiddleston', 'Tom_Holland', 'Tuppence_Middleton',
    #    'Ursula_Corbero', 'Wentworth_Miller', 'Zac_Efron', 'Zendaya',
    #    'Zoe_Saldana', 'alycia_dabnem_carey', 'amber_heard',
    #    'barack_obama', 'barbara_palvin', 'camila_mendes',
    #    'elizabeth_olsen', 'ellen_page', 'elon_musk', 'gal_gadot',
    #    'grant_gustin', 'jeff_bezos', 'kiernen_shipka', 'margot_robbie',
    #    'melissa_fumero', 'scarlett_johansson', 'tom_ellis']

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])
        # label = self.labels.index(self.data[index].split("/")[-1][:-9])
        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        edge = self.load_edge(img_gray, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        # return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask), label
        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            # print(canny(img, sigma=sigma, mask=mask).astype(np.float))
            # imsave("./test_samples/{}.png".format(index), canny(img, sigma=sigma, mask=mask).astype(np.float))
            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
