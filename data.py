
import torch.utils.data as data

from PIL import Image
import os
import os.path
import json

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


Image.MAX_IMAGE_PIXELS = 1e9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def image_to_tensor_PIL(path, image_size):
    try:
        image = Image.open(path).convert('RGB')
    except:
        print(f'cropt image:{path}, then remove')
        # shutil.move(path, './')
    # convert a PIL image to tensor (HWC) in range [0,255] to a torch.Tensor(CHW) in the range [0.0,1.0]
    # data_transform = transform.Compose([transform.Resize((image_size, image_size)), transform.ToTensor(),
    #                                     transform.Normalize(mean, std)])
    while(min(image.size[0], image.size[1]))<256:
        image = image.resize((image.size[0]*2, image.size[1]*2))

    alpha=max(image.size[0], image.size[1])/min(image.size[0], image.size[1])
    while(max(image.size[0], image.size[1])>1800 and alpha <3.5):
        image = image.resize((image.size[0]//2, image.size[1]//2))
    # data_transform = transform.Compose([transform.Resize((512, 512)), transform.CenterCrop((image_size, image_size)), transform.ToTensor(),
    #                                     transform.Normalize(mean, std)])
    data_transform = transform.Compose([transform.CenterCrop((image_size, image_size)), transform.ToTensor(),
                                        transform.Normalize(mean, std)])
    image_tensor = data_transform(image)

    return image_tensor
def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def make_dataset2(json_dir):
    with open(json_dir, 'r', encoding='utf_8') as fp:
        image_paths = json.load(fp)
    return image_paths


def default_loader(path, flag, image_size):
    if flag:
        return Image.open(path).convert('RGB').resize([image_size,image_size])
    else:
        return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, path, transform=None, return_paths=False,
                 loader=default_loader, resize=False):
        imgs = sorted(make_dataset(path))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + path + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))


        # self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.flag = resize
        self.style_size = 256
        #self.content_size =

    def __getitem__(self, index):
        path = self.imgs[index]
        print(path)
        if self.flag:
            img = Image.open(path).convert('RGB')
            #img = img.resize((img.size[0]//2, img.size[1]//2))
            img = img.resize((256,256))
        else:
            img = Image.open(path).convert('RGB')
        #img = self.loader(path, self.flag)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = image_to_tensor_PIL(path, self.style_size)
        return img, path

    def __len__(self):
        return len(self.imgs)

