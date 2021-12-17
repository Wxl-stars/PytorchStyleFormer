import cv2

path = '/data2/wuxiaolei/compare-model-pair/adain/input/mask/mask.png'
img = cv2.imread(path)
img2 = 255 - img
cv2.imwrite('mask2.png',img2)
print(img2)
