from functions import *
import  matplotlib.pyplot as plt
import os


def count_positive(lst):
    return sum(1 for x in lst if x >= 0)

def find_indices_of_positive_elements(lst):
    return [index for index, element in enumerate(lst) if element >= 0]

def cal_multi_data(input_dir, img_threshold, intensity_threshold1, intensity_threshold2, result_path):
    name = input_dir.split('/')[-1].split('.')[0]
    n1 = '%s_8_gaussian' % name
    n2 = '%s_9_gaussian' % name
    n3 = '%s_10_gaussian' % name
    n4 = '%s_11_gaussian' % name
    r1, img_line1_1, img_line2_1 = cal_data(n1, img_threshold)
    r2, img_line1_2, img_line2_2 = cal_data(n2, img_threshold)
    r3, img_line1_3, img_line2_3 = cal_data(n3, img_threshold)
    r4, _, _ = cal_data(n4, img_threshold)

    rs = [r1, r2, r3]
    # print(rs)
    # print(r4)

    # img_lines1 = [img_line1_1, img_line1_2, img_line1_3]
    img_lines2 = [img_line2_1, img_line2_2, img_line2_3]

    li = [r1-intensity_threshold1, r2-intensity_threshold1, r3-intensity_threshold1]
    num = count_positive(li)

    max_index = rs.index(max(rs))

    if num != 0:
        print('Tumor is found!')
        # img_line1 = img_lines1[max_index]
        img_line2 = img_lines2[max_index]

        plt.imshow(img_line2, cmap='Greys_r')
        plt.axis('off')
        # plt.text(100, 30, 'Tumor is found!', fontsize=15, horizontalalignment='center', color='white')
        # print(max_index)
        # print(rs[max_index])
        # plt.savefig(result_path + 'result.png', bbox_inches='tight', pad_inches=0.0, dpi=600)
        plt.show()
        plt.close()
        io.imsave(result_path + 'result.png', img_line2, check_contrast=False)

        # plt.imshow(img_line1)
        # plt.axis('off')
        # plt.savefig(result_path + 'result2.png', bbox_inches='tight', pad_inches=0.0, dpi=600)
        # plt.show()
        # plt.close()

    else:
        r_max = rs[max_index]
        ratio = r4 / r_max
        if ratio >= intensity_threshold2:
            print('Tumor is found!')
            img_line2 = img_lines2[max_index]
            plt.imshow(img_line2, cmap='Greys_r')
            plt.axis('off')
            # plt.text(100, 30, 'Tumor is found!', fontsize=15, horizontalalignment='center', color='white')
        
            # plt.savefig(result_path + 'result.png', dpi=600)
            plt.show()
            plt.close()
            io.imsave(result_path + 'result.png', img_line2, check_contrast=False)

        else:
            print('There is no tumor!')


if __name__ == '__main__':

    input_dir = './data/image.tif'

    img_threshold = 0.3
    intensity_threshold1 = 1.259
    intensity_threshold2 = 1.254

    result_path = './results/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    cal_multi_data(input_dir, img_threshold, intensity_threshold1, intensity_threshold2, result_path)
