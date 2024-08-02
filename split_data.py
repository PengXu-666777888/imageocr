import numpy as np
from skimage import io, filters
import os


def gaussian(pic_raw):
	pic = filters.gaussian(pic_raw,sigma=3)

	return pic

result_path = './image1/'
if not os.path.exists(result_path):
    os.mkdir(result_path)
    

input_dir = './data/image.tif'
img = io.imread(input_dir)
id = input_dir.split('/')[-1].split('.')[0]

# 图8
img_origin = img[60:360, 20:320, :]

io.imsave(result_path + '%s_8_origin.tif' % id, img_origin, check_contrast=False)

img_gau = gaussian(img_origin[:, :, 1])
print(img_gau.shape)
img_out = np.zeros([300, 300, 3], dtype=np.float32)
img_out[:, :, 1] = img_gau

io.imsave(result_path + '%s_8_gaussian.tif' % id, img_out, check_contrast=False)

# 图9
img_origin = img[60:360, 320:620, :]

io.imsave(result_path + '%s_9_origin.tif' % id, img_origin, check_contrast=False)

img_gau = gaussian(img_origin[:, :, 1])
print(img_gau.shape)
img_out = np.zeros([300, 300, 3], dtype=np.float32)
img_out[:, :, 1] = img_gau

io.imsave(result_path + '%s_9_gaussian.tif' % id, img_out, check_contrast=False)

# 图10
img_origin = img[60:360, 620:920, :]

io.imsave(result_path + '%s_10_origin.tif' % id, img_origin, check_contrast=False)

img_gau = gaussian(img_origin[:, :, 1])
print(img_gau.shape)
img_out = np.zeros([300, 300, 3], dtype=np.float32)
img_out[:, :, 1] = img_gau

io.imsave(result_path + '%s_10_gaussian.tif' % id, img_out, check_contrast=False)

# 图11
img_origin = img[60:360, 920:1220, :]

io.imsave(result_path + '%s_11_origin.tif' % id, img_origin, check_contrast=False)

img_gau = gaussian(img_origin[:, :, 1])
print(img_gau.shape)
img_out = np.zeros([300, 300, 3], dtype=np.float32)
img_out[:, :, 1] = img_gau

io.imsave(result_path + '%s_11_gaussian.tif' % id, img_out, check_contrast=False)

