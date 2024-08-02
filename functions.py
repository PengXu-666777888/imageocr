import numpy as np
from skimage import measure
from skimage import io, filters
import  matplotlib.pyplot as plt


def normalization(image):
	"""
	归一化函数（线性归一化）
	:param image: ndarray
	:return:
	"""
	# as dtype = np.float32
	image = image.astype(np.float32)
	image = (image - np.min(image)) / (np.max(image)-np.min(image)+ 1e-14)
	
	return image

def clean_data(image, bright_threshold=0.38):
	image[image<bright_threshold] = 0
	
	return image

def binaryzation(image):
	pic = image[:, :, 1]
	a = np.nonzero(pic)
	
	a_h, a_w = a[0], a[1]
	length = len(a_h)
	c = np.zeros([300, 300, 3], dtype=np.float32)
	for i in range(length):
		c[a_h[i], a_w[i], 1] = 1

	return c

def find_nonzero_boundaries(image):
    # 获取矩阵的尺寸
    image = image[:, :, 1]
    rows, cols = image.shape
 
    # 初始化边界列表
    boundaries = []
 
    # 检查每一行和每一列
    for i in range(rows):
        for j in range(cols):
            # 如果当前元素非零，则检查其上下左右四个方向
            if image[i, j] != 0:
                # 上方、下方、左方、右方
                if (i > 0 and i < rows - 1 and image[i - 1, j] != 0 and image[i + 1, j] != 0) and (j > 0 and j < cols - 1 and image[i, j - 1] != 0 and image[i, j + 1] != 0):
                    continue
				
                # 如果四个方向都没有非零元素，则当前元素是一个边界
                boundaries.append((i, j))
 
    return boundaries

# def fill_binaryzation(image, index_list, indexs):
#      m, n, _ = image.shape
#      for ind in indexs:
#             index_list2 = []
    
#             for l in index_list:
#                 if ind[0] <= l[0] <= ind[2] and ind[1] <= l[1] <= ind[3]:
#                     index_list2.append(l) 
                    
#             for l in index_list2:
#                 min_row, min_col, max_row, max_col = np.min(l[0]), np.min(l[1]), np.max(l[0]), np.max(l[1])
#                 # print(min_row, min_col, max_row, max_col)
#                 for i in range(m):
#                     for j in range(n):
#                             if min_row <= i <= max_row and min_col <= j <= max_col:
#                                 image[i, j, 1] = 1
#      return image

def fill_binaryzation(image, indexs, s=2):
    for ind in indexs:
        img = image[max(0, ind[0]):min(300, ind[2]+1), max(0, ind[1]):min(300, ind[3]+1), 1]
        m, n = img.shape
        k1 = m+2*s
        k2 = n+2*s
        d1 = np.zeros([k1, k2], dtype=np.float32)
        d1[s:-s, :-2*s] = img[:, :]
        d2 = np.zeros([k1, k2], dtype=np.float32)
        d2[s:-s, 2*s:] = img[:, :]
        d3 = np.zeros([k1, k2], dtype=np.float32)
        d3[:-2*s, s:-s] = img[:, :]
        d4 = np.zeros([k1, k2], dtype=np.float32)
        d4[2*s:, s:-s] = img[:, :]

        d = d1 + d2 + d3 + d4

        pic = d[s:-s, s:-s]
        a = np.nonzero(pic)
        a_h, a_w = a[0], a[1]
        length = len(a_h)
        c = np.zeros([m, n], dtype=np.float32)
        for i in range(length):
            c[a_h[i], a_w[i]] = 1
        
        image[max(0, ind[0]):min(300, ind[2]+1), max(0, ind[1]):min(300, ind[3]+1), 1] = c
     
    return image

def fill_binaryzation2(image, s=1):

    m, n = image.shape
    k1 = m+2*s
    k2 = n+2*s
    d1 = np.zeros([k1, k2], dtype=np.float32)
    d1[s:-s, :-2*s] = image[:, :]
    d2 = np.zeros([k1, k2], dtype=np.float32)
    d2[s:-s, 2*s:] = image[:, :]
    d3 = np.zeros([k1, k2], dtype=np.float32)
    d3[:-2*s, s:-s] = image[:, :]
    d4 = np.zeros([k1, k2], dtype=np.float32)
    d4[2*s:, s:-s] = image[:, :]

    d = d1 + d2 + d3 + d4

    pic = d[s:-s, s:-s]
    a = np.nonzero(pic)
    a_h, a_w = a[0], a[1]
    length = len(a_h)
    c = np.zeros([m, n], dtype=np.float32)
    for i in range(length):
        c[a_h[i], a_w[i]] = 1
    
    return c

def zone_division(index_list):
    d = np.zeros([300, 300], dtype=np.float32)
    for l in index_list:
        d[l[0], l[1]] = 1
    label_img = measure.label(d)
    # print("区域数量:", len(np.unique(label_img)))

    indexs = []
    for region in measure.regionprops(label_img):
        min_row, min_col, max_row, max_col = region.bbox
        indexs.append((min_row, min_col, max_row, max_col))
    
    return indexs

def del_area(image, indexs):
    for ind in indexs:
        num=0
        for i in range(max(0, ind[0]), min(300, ind[2]+1)):
            for j in range(max(0, ind[1]), min(300, ind[3]+1)):
                if image[i, j , 1] == 1:
                    num += 1
        if num < 100:
             image[max(0, ind[0]):min(300, ind[2]+1), max(0, ind[1]):min(300, ind[3]+1), 1] = 0
    
    return image

def cal_area(image, indexs):
    nums = []
    for ind in indexs:
        num=0
        for i in range(max(0, ind[0]), min(300, ind[2]+1)):
            for j in range(max(0, ind[1]), min(300, ind[3]+1)):
                if image[i, j , 1] == 1:
                    num += 1
        nums.append(num)

    return nums

def cal_intensity(image, image2, indexs):
    nums = []
    for ind in indexs:
        a1 = image[max(0, ind[0]):min(300, ind[2]+1), max(0, ind[1]):min(300, ind[3]+1)]
        a2 = image2[max(0, ind[0]):min(300, ind[2]+1), max(0, ind[1]):min(300, ind[3]+1)]
        a = a1 * a2
        nums.append(np.mean(a))
        
    return nums

def cal_intensity_ratio(intensitys, areas):
    length = len(intensitys)
    means = []
    for i in range(length):
        intensity_mean = intensitys[i] / areas[i]
        means.append(intensity_mean)
    
    return means[1]/means[0]

def cal_finally_list(index_list, indexs, i):
    # print(indexs)
    
    ind = indexs[i]
    index_list2 = []
    
    for l in index_list:
        if ind[0] <= l[0] <= ind[2] and ind[1] <= l[1] <= ind[3]:
            index_list2.append(l) 
        
    return index_list2

def drawing_line(image, index_list, indexs, i):
    index_list2 = cal_finally_list(index_list, indexs, i)
    for l in index_list2:
        image[l[0], l[1], 0] = 1
        image[l[0], l[1], 1] = 0
    
    return image

def drawing_line2(index_list, indexs, i):
    index_list2 = cal_finally_list(index_list, indexs, i)
    image = np.zeros([300, 300], dtype=np.uint8)
    for l in index_list2:
        image[l[0], l[1]] = 255
    
    return image

def trans(id):
    a = id.split('_')[0]
    b = id.split('_')[1]

    if len(b) == 1:
        b = '0' + b

    return a + '_' + b

def processing_data(img_norm, threshold):
    # 数据清理后的结果
    img_clean = clean_data(img_norm, threshold)

    # 二值化后的结果
    img_binary = binaryzation(img_clean)

    # 得到边界坐标，并保存为列表
    index_list = find_nonzero_boundaries(img_binary)

    # 划分区域，得到区域的坐标范围（min_row, min_col, max_row, max_col），将多个区域保存为一个列表
    indexs = zone_division(index_list)

    # 消除较小区域
    img_del_binary = del_area(img_binary, indexs)

    # 对消除后的结果进行填充
    img_fill_binary = fill_binaryzation(img_del_binary, indexs)

    # 重新得到填充后的边界坐标，并保存为列表
    index_list2 = find_nonzero_boundaries(img_fill_binary)

    # 重新划分填充后的区域，得到区域的坐标范围（min_row, min_col, max_row, max_col），将多个区域保存为一个列表
    indexs2 = zone_division(index_list2)

    return index_list2, indexs2, img_fill_binary, img_clean

def cal_data(input_name, threshold):
    threshold2 = 0.5
    p_threshold = 0.25
    # 读取数据
    img = io.imread('./image1/%s.tif' % input_name)
    img1 = img.copy()
    img *= 255

    # 得到名字
    id = input_name.split('_')[0] + '_' + input_name.split('_')[1]

    # 读取原始裁切数据
    img_origin = io.imread('./image1/%s_origin.tif' % id)

    # 归一化后的结果
    img_norm = normalization(img)
    img_norm_copy = img_norm.copy()

    index_list, indexs, img_fill_binary, img_clean = processing_data(img_norm, threshold)
    index_list2, indexs2, _, _ = processing_data(img_norm_copy, threshold2)

    # 计算各区域面积，将多个区域面积保存为一个列表
    areas = cal_area(img_fill_binary, indexs)
    i_max = areas.index(max(areas))

    index = indexs[i_max]

    imgc = img_clean[index[0]:index[2]+1, index[1]:index[3]+1, 1]
    imgo = img_origin[index[0]:index[2]+1, index[1]:index[3]+1, 1]
    imgo_copy = imgo.copy()
    imgo_copy[imgc<threshold2] = 0
    num0 = np.count_nonzero(imgo_copy)

    imgc[imgc>=threshold2] = 0
    # imgc = fill_binaryzation2(imgc)
    # imgg[imgg<0.3] = 0
    # c[index[0]:index[2]+1, index[1]:index[3]+1, :] = imgg
    
    m, n = imgc.shape
    
    for i in range(m):
        for j in range(n):
            if imgc[i, j] == 0:
                imgo[i, j] = 0
    imgc1 = imgc[:m//2, :n//2]
    imgo1 = imgo[:m//2, :n//2]
    int1 = np.sum(imgo1)
    # print(int1)
    num1 = np.count_nonzero(imgc1)
    
    imgc2 = imgc[:m//2, n//2:]
    imgo2 = imgo[:m//2, n//2:]
    int2 = np.sum(imgo2)
    # print(int2)
    num2 = np.count_nonzero(imgc2)

    imgc3 = imgc[m//2:, :n//2]
    imgo3 = imgo[m//2:, :n//2]
    int3 = np.sum(imgo3)
    # print(int3)
    num3 = np.count_nonzero(imgc3)

    imgc4 = imgc[m//2:, n//2:]
    imgo4 = imgo[m//2:, n//2:]
    int4 = np.sum(imgo4)
    # print(int4)
    num4 = np.count_nonzero(imgc4)

    p1 = num1 / (num1+num2+num3+num4)
    p2 = num2 / (num1+num2+num3+num4)
    p3 = num3 / (num1+num2+num3+num4)
    p4 = num4 / (num1+num2+num3+num4)

    # print(p1)
    # print(p2)
    # print(p3)
    # print(p4)

    if p1 < p_threshold:
        p1 = 0
        imgc[:m//2, :n//2] = 0
    else:
        p1 = 1

    if p2 < p_threshold:
        p2 = 0
        imgc[:m//2, n//2:] = 0
    else:
        p2 = 1

    if p3 < p_threshold:
        p3 = 0
        imgc[m//2:, :n//2] = 0
    else:
        p3 = 1

    if p4 < p_threshold:
        p4 = 0
        imgc[m//2:, n//2:] = 0
    else:
        p4 = 1

    int_sum = (int1 * p1 + int2 * p2 + int3 * p3 + int4 * p4) / (num1 * p1 + num2 * p2 + num3 * p3 + num4 * p4)
    int_sum2 = np.sum(imgo_copy)/ num0
    # print(int_sum)
    # print(int_sum2)
    result = round(int_sum2/int_sum, 2)

    # c = np.zeros([m, n, 3], dtype=np.float32)
    # c[:, : , 1] = imgc
    # io.imsave('%s_clean.tif' % id, c, check_contrast=False)


    # 返回线的图
    i = areas.index(max(areas))
    # print(i)
    # print(indexs)
    img_line = drawing_line(img1, index_list2, indexs2, i)
    img_line2 = drawing_line2(index_list2, indexs2, i)


    return result, img_line, img_line2
