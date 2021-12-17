import csv
import imghdr

import numpy as np
import os
import cv2

import pandas
import torch
from PIL import Image
from tqdm import tqdm
import scipy.misc
import random
import torchvision.transforms as transform

# from utils import image_check, get_one_hot_encoded_vector
from utils import image_check, normalize_vgg


def get_batch_tensor(dataset, batch_size=2):
    batch_tensor = None
    for i in range(batch_size):
        index = random.randint(1, len(dataset))
        image_tensor = dataset.getitem(index)
        image_tensor.unsqeeze_(0)
        if batch_tensor is None:
            batch_tensor = image_tensor
        else:
            batch_tensor = torch.cat((batch_tensor, image_tensor), 0)
    return batch_tensor


def image_to_tensor(image, image_size):
    image_tensor = torch.from_numpy(
        cv2.cvtColor(cv2.resize(image, dsize=(image_size, image_size)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)  # C,H,W
    # image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    return image_tensor


def image_to_tensor_PIL(path, image_size):
    image = Image.open(path).convert('RGB')
    # image = Image.open(path).convert('RGB').resize(image_size, image_size)
    # convert a PIL image to tensor (HWC) in range [0,255] to a torch.Tensor(CHW) in the range [0.0,1.0]
    data_transform = transform.Compose([transform.Resize((image_size, image_size)), transform.ToTensor(),
                                        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_tensor = data_transform(image)
    return image_tensor


class ArtDataset():
    def __init__(self, opts):
        self.path_to_art_dataset = opts.art_data_path
        self.style_image_paths = []

        self.image_size = opts.image_size
        # self.dataset = [os.path.join(self.path_to_art_dataset, x) for x in os.listdir(self.path_to_art_dataset)]
        print("Art dataset contains %d images." % len(self.style_image_paths))

    def __len__(self):
        return len(self.style_image_paths) // 2

    def __getitem__(self, item):
        path, pathb = self.style_image_paths[item * 2], self.style_image_paths[item * 2 + 1]
        image_tensor1 = image_to_tensor(path, self.image_size)
        image_tensor2 = image_to_tensor(pathb, self.image_size)
        # label1 =
        return image_tensor1, image_tensor2


def find_index(list, data):
    for i, item in enumerate(list):
        if item == data:
            return i
    return None


def preprocess(path_to_art_dataset, info_path):
    style_image_paths = []
    style_str = []
    style_num = []
    label = []
    with open(info_path, 'r') as rfile:
        reader = csv.DictReader(rfile)
        for row in reader:
            if row['style'] is not '':
                style_image_paths.append(os.path.join(path_to_art_dataset, row['filename']))
                style_str.append(row['style'])
    label = list(set(style_str))
    for i, data in enumerate(style_str):
        style_num.append(find_index(label, data))


class StyleDataset():
    def __init__(self, opts):
        self.image = []
        self.style = []
        self.root = opts.art_data_path
        self.image_size = opts.image_size
        with open(opts.csv_path, 'r') as rfile:
            reader = csv.DictReader(rfile)
            self.image = [row['filename'] for row in reader]
            self.style = [row['style'] for row in reader]
            label = list(set(self.style))
            for i, data in enumerate(self.style):
                self.style[i] = find_index(label, data)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        path = os.path.join(self.root, self.image[item])
        image_tensor = image_to_tensor(path, self.image_size)
        label = self.style[item]
        return image_tensor, label


class PlacesDataset():
    categories_names = ['/a/alley', '/a/apartment_building/outdoor',
                        '/a/aqueduct', '/a/arcade', '/a/arch',
                        '/a/atrium/public', '/a/auto_showroom',
                        '/a/amphitheater',
                        '/b/balcony/exterior', '/b/balcony/interior', '/b/badlands',
                        '/b/ballroom', '/b/banquet_hall',
                        '/b/bar', '/b/barn',
                        '/b/bazaar/outdoor', '/b/beach', '/b/beach_house',
                        '/b/bedroom', '/b/beer_hall',
                        '/b/boat_deck', '/b/bookstore', '/b/botanical_garden',
                        '/b/bridge', '/b/bullring',
                        '/b/building_facade', '/b/butte',
                        '/c/cabin/outdoor', '/c/campsite', '/c/campus', '/c/canal/natural',
                        '/c/canyon', '/c/canal/urban',
                        '/c/carrousel', '/c/castle', '/c/chalet',
                        '/c/church/indoor', '/c/church/outdoor',
                        '/c/cliff', '/c/crevasse', '/c/crosswalk',
                        '/c/coast', '/c/coffee_shop',
                        '/c/corn_field', '/c/corral',
                        '/c/courthouse', '/c/courtyard',
                        '/d/desert/sand', '/d/desert_road'
                                          '/d/doorway/outdoor', '/d/downtown',
                        '/d/dressing_room',
                        '/e/embassy', '/e/entrance_hall',
                        '/f/field/cultivated', '/f/field/wild',
                        '/f/field_road',
                        '/f/formal_garden', '/f/florist_shop/indoor',
                        '/f/fountain', '/g/gazebo/exterior',
                        '/g/general_store/outdoor', '/g/glacier',
                        '/g/grotto',
                        '/h/harbor', '/h/hayfield',
                        '/h/hotel/outdoor',
                        '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_floe',
                        '/i/iceberg', '/i/igloo',
                        '/i/inn/outdoor', '/i/islet', '/j/junkyard', '/k/kasbah',
                        '/l/lagoon',
                        '/l/lake/natural', '/l/lawn',
                        '/l/legislative_chamber', '/l/library/outdoor', '/l/lighthouse',
                        '/l/lobby', '/m/mansion',
                        '/m/marsh', '/m/mausoleum',
                        '/m/moat/water', '/m/mosque/outdoor',
                        '/m/mountain_path', '/m/mountain_snowy', '/m/museum/outdoor',
                        '/o/oast_house', '/o/ocean', '/o/orchestra_pit', '/p/pagoda',
                        '/p/palace',
                        '/p/pasture', '/p/phone_booth',
                        '/p/picnic_area', '/p/pizzeria',
                        '/p/plaza', '/p/pond',
                        '/r/racecourse', '/r/restaurant_patio', '/r/rice_paddy', '/r/river',
                        '/r/ruin',
                        '/s/schoolhouse',
                        '/s/shopfront', '/s/shopping_mall/indoor',
                        '/s/ski_resort', '/s/sky', '/s/street',
                        '/s/stable', '/s/swimming_hole', '/s/synagogue/outdoor', '/t/temple/asia',
                        '/t/throne_room', '/t/tower',
                        '/t/tree_house', '/t/tundra', '/v/valley',
                        '/v/viaduct', '/v/village', '/v/volcano',
                        '/w/water_park', '/w/waterfall',
                        '/w/wave', '/w/wheat_field', '/w/wind_farm',
                        '/w/windmill', '/y/yard',
                        ]
    categories_names = [x[1:] for x in categories_names]

    def __init__(self, opts, augmentor):
        self.path_to_dataset = opts.content_data_path
        self.content_image_size = opts.image_size
        self.content_image_paths = []
        self.categories = []
        self.aug = augmentor
        for category_idx, category_name in enumerate(tqdm(self.categories_names, ncols=100, mininterval=.5)):
            # print(category_name, category_idx)
            if os.path.exists(os.path.join(self.path_to_dataset, category_name)):
                for file_name in os.listdir(os.path.join(self.path_to_dataset, category_name)):
                    self.content_image_paths.append(os.path.join(self.path_to_dataset, category_name, file_name))
                    self.categories.append(category_name)
            else:
                pass
                # print("Category %s can't be found in path %s. Skip it." %
                #       (category_name, os.path.join(path_to_dataset, category_name)))

        print("Finished. Constructed Places2 dataset of %d images." % len(self.content_image_paths))

        self.path_to_art_dataset = opts.art_data_path  # train_image
        self.artist_list = os.listdir(self.path_to_art_dataset)
        self.style_image_paths = []
        self.artist_slugs = []
        self.artist_slugs_oneshot = []
        self.classes = []

        self.image_size = opts.image_size

        for category_idx, artist_slug in enumerate(tqdm(self.artist_list)):
            for file_name in tqdm(os.listdir(os.path.join(self.path_to_art_dataset, artist_slug))):
                self.style_image_paths.append(os.path.join(self.path_to_art_dataset, artist_slug, file_name))
                self.artist_slugs.append(artist_slug)
        print("Art dataset contains %d images." % len(self.style_image_paths))

    def __len__(self):
        return min(len(self.content_image_paths) // 2, len(self.style_image_paths) // 2)

    def __getitem__(self, item):
        content_index = random.randint(0, len(self.content_image_paths) - 1)
        content_path = self.content_image_paths[content_index]

        style_index = random.randint(0, len(self.style_image_paths) - 1)
        style_path = self.style_image_paths[style_index]
        content_image = cv2.imread(content_path)
        style_image = cv2.imread(style_path)
        content_image = image_check(content_image, augmentor=self.aug)
        style_image = image_check(style_image, augmentor=self.aug)

        content_tensor = normalize_vgg(image_to_tensor(content_image, self.image_size))
        style_tensdor = normalize_vgg(image_to_tensor(style_image, self.image_size))

        return content_tensor, style_tensdor

    def get_label_num(self):
        return len(self.artist_list)

    def get_image_num(self):
        return len(self.content_image_paths), len(self.style_image_paths)

    def get_label_lenth(self):
        return len(self.artist_list), len(self.categories)


