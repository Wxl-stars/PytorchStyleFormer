
from PIL import Image


# def preprocess(content_path):
def preprocess(img):
    # img = Image.open(content_path).convert('RGB')
    H, W = img.shape

    if max(H, W) >= 256 and max(H, W) < 512:
        long_side = 256
    if max(H, W) >= 512 and max(H, W) < 1024:
        long_side = 512
    if max(H, W) > 1024:
        long_side = 1024

    if H > W:
        H = long_side
        W = int(long_side / H * W)
    return H, W

