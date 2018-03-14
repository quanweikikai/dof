#! /usr/bin/python

import numpy as np
import cv2
import glob
from multiprocessing import Process
import matplotlib.pyplot as plt

size_list = [(6000,4000),(2400,1600),(960,640),(480,320)]
#zoom_rate = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
zoom_rate = 2
# normal filter
kernel_normal = np.matrix([1,4,6,4,1])
kernel_normal = kernel_normal.T*kernel_normal/256.
# smallest image filter
smallest_kernel = np.matrix([3,3,4,3,3])
smallest_kernel = smallest_kernel.T*smallest_kernel/256.
# DCT minus filter
dct_minus_kernel_1 = np.matrix([26,-10,-32,-10,26])
dct_minus_kernel_2 = np.matrix([-26,10,32,10,-26])
dct_minus_kernel = dct_minus_kernel_1.T*dct_minus_kernel_2/64./64.
# DCT plus filter
dct_plus_kernel_1 = np.matrix([26,-10,-32,-10,26])
dct_plus_kernel_2 = np.matrix([26,-10,-32,-10,26])
dct_plus_kernel = dct_plus_kernel_1.T*dct_plus_kernel_2/64./64.
# DCT bokeh detection zoom size
dct_boke_detection_h_size = 640
dct_boke_detection_v_size = 480

#-----------___________________-----------#
#-----------\     Layer 0     /-----------#
#------------\_______________/------------#
#-------------\   Layer 1   /-------------#
#--------------\___________/--------------#
#---------------\ Layer 2 /---------------#
#----------------\_______/----------------#
#---------------- Layer 3-----------------# black and white(Y only) after Layer 3
#----------------   ...  -----------------# 

#calculate pyramid size-------------------------------------
full_size = (6000,4000)
size_list = [full_size]
min_size_lim = 200
pyramid_size = 10
sub_size = full_size
for i in xrange(1, 100):
    sub_size = (int(sub_size[0]/zoom_rate)&~3, int(sub_size[1]/zoom_rate)&~3)
    size_list.append(sub_size)
    if sub_size[0] < min_size_lim:
        pyramid_size = i
        break
#end of pyramid size calculation

def noise_reduction(image, dct_thresh = 30):
    assert len(image.shape) == 2
    original_size = image.shape
    image = cv2.resize(image, (dct_boke_detection_h_size, dct_boke_detection_v_size))
    image = image.astype(np.int16)
    dct_plus_img = cv2.filter2D(image, -1, dct_plus_kernel)
    dct_minus_img = cv2.filter2D(image, -1, dct_minus_kernel)
    dct_plus_img[dct_plus_img<0] = 0
    dct_minus_img[dct_minus_img<0] = 0
    dct_img = dct_plus_img + dct_minus_img
    dct_img[dct_img<dct_thresh] = 0
    dct_img[dct_img>=dct_thresh] = 1
    dct_result_img = cv2.resize(dct_img, (original_size[1], original_size[0]))
    return dct_result_img

def make_pyramid(image, Y_layer=3):
    pyramid_list = []
    image = image.astype(np.int16)
    for i in xrange(len(size_list)-1):
        img_size = size_list[i]
        sub_size = size_list[i+1]
        image_sub = cv2.resize(image, sub_size)
        if (i + 1 >= Y_layer):
            image_sub[:,:,1:] = 128
            #plt.imshow(cv2.cvtColor(image_sub.astype(np.uint8), cv2.COLOR_YUV2BGR))
            #plt.show()
        image_enlarge = cv2.resize(image_sub, img_size)
        img_plus = image - image_enlarge
        img_minus = image_enlarge - image
        img_plus[img_plus<0] = 0
        img_minus[img_minus<0] = 0
        pyramid_list.append(img_plus)
        pyramid_list.append(img_minus)
        image = image_sub
    #pyramid bottom image
    image = image.astype(np.uint8)

    return pyramid_list,image

def make_map(pyramid_list):
    map_list = []
    for i in xrange(len(pyramid_list)/2):
        # merge 2 edges
        map_merged = (pyramid_list[2*i] + pyramid_list[2*i+1])
        # resize merged edge
        original_size = map_merged.shape
        zoom_size = (int(original_size[1]/zoom_rate)&~3, int(original_size[0]/zoom_rate)&~3)
        map_merged = map_merged*3
        resized_map = cv2.resize(map_merged, zoom_size)
        for j in xrange(4):
            resized_map = cv2.filter2D(resized_map, -1, kernel_normal)
        map_merged = cv2.resize(resized_map, (original_size[1], original_size[0]))
        map_merged[map_merged<0] = 0
        map_merged[map_merged>255] = 255
        map_list.append(map_merged)
    return map_list

def remap_image(pyramid_list1, pyramid_list2, map_list1, map_list2):
    assert len(map_list1) == len(map_list2)
    assert len(pyramid_list1) == len(pyramid_list2)
    assert len(pyramid_list1) == 2 * len(map_list1)
    remap_list = []
    map_list = []
    for i in xrange(len(map_list1)):
        img_minus = np.zeros(pyramid_list1[i*2+1].shape, dtype=np.uint8)
        img_plus = np.zeros(pyramid_list1[i*2].shape, dtype=np.uint8)
        img_map = np.zeros(map_list1[i].shape, dtype=np.uint8)
        idx_1 = map_list1[i] > map_list2[i]
        idx_2 = map_list1[i] <= map_list2[i]
        img_plus[idx_1] = pyramid_list1[i*2][idx_1]
        img_plus[idx_2] = pyramid_list2[i*2][idx_2]
        img_minus[idx_1] = pyramid_list1[i*2+1][idx_1]
        img_minus[idx_2] = pyramid_list2[i*2+1][idx_2]

        img_map[idx_1] = map_list1[i][idx_1]
        img_map[idx_2] = map_list2[i][idx_2]
        remap_list.append(img_plus)
        remap_list.append(img_minus)
        map_list.append(img_map)
    return remap_list, map_list

def remap_min_image(min1,min2):
    assert min1.shape == min2.shape
    min1 = min1.astype(np.int16)
    min2 = min2.astype(np.int16)
    min_blur1 = cv2.filter2D(min1, -1, smallest_kernel)
    min_blur2 = cv2.filter2D(min2, -1, smallest_kernel)
    min_map1 = np.abs(min1-min_blur1)[:,:,0]
    min_map2 = np.abs(min2-min_blur2)[:,:,0]
    #plt.imshow(min_map1,cmap="gray")
    #plt.show()
    #min_map1 = cv2.filter2D(min_map1, -1, kernel)
    #min_map2 = cv2.filter2D(min_map2, -1, kernel)
    # resize merged edge
    original_size = min1.shape
    min_map1 = min_map1*3
    min_map2 = min_map2*3
    zoom_size = (int(original_size[1]/zoom_rate)&~3, int(original_size[0]/zoom_rate)&~3)
    resized_map_1 = cv2.resize(min_map1, zoom_size)
    resized_map_2 = cv2.resize(min_map2, zoom_size)
    for j in xrange(4):
        resized_map_1 = cv2.filter2D(resized_map_1, -1, kernel_normal)
        resized_map_2 = cv2.filter2D(resized_map_2, -1, kernel_normal)
    min_map1 = cv2.resize(resized_map_1, (original_size[1], original_size[0]))
    min_map2 = cv2.resize(resized_map_2, (original_size[1], original_size[0]))
    #compare
    idx_1 = min_map1 > min_map2
    idx_2 = min_map1 <= min_map2
    remapped_img = np.zeros(min1.shape, dtype=np.uint8)
    remapped_img[idx_1] = min1[idx_1]
    remapped_img[idx_2] = min2[idx_2]
    remapped_img = remapped_img.astype(np.uint8)
    #cv2.imshow("result",remapped_img)
    #cv2.waitKey(0)
    return remapped_img

    
def merge_all(min_img, pyramid_list):
    merged_img = min_img.astype(np.int16)
    #kernel = np.matrix([1,4,6,4,1])
    #kernel = kernel.T*kernel/256.
    for i in xrange(len(pyramid_list)/2):
        merged_img = cv2.resize(merged_img, size_list[-2-i])
        #merged_img = cv2.filter2D(merged_img, -1, kernel)
        #pay attention: merge image inversely(from back to front)
        merged_img -= pyramid_list[-i*2-1]
        merged_img += pyramid_list[-i*2-2]
    merged_img[merged_img>255] = 255
    merged_img[merged_img<0] = 0
    merged_img = merged_img.astype(np.uint8)
    return merged_img

image_folder = "./group2/"
def main():
    image_name = sorted(glob.glob(image_folder+"*.JPG"))
    image1 = cv2.imread(image_name[0])
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
    pyramid_list_base, min_base = make_pyramid(image1)
    map_list_base = make_map(pyramid_list_base)
    for i in xrange(1,len(image_name)):
        print "process No.%d image"%(i)
        image2 = cv2.imread(image_name[i])
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2YUV)
        assert image1.shape == image2.shape
        # noise reduction offset
        offset = noise_reduction(image2[:,:,0])
        # apply offset to plus and minus edge at the top of pyramid
        pyramid_list2, min2 = make_pyramid(image2)
        pyramid_list2[0][:,:,0] = pyramid_list2[0][:,:,0] - offset
        pyramid_list2[1][:,:,0] = pyramid_list2[1][:,:,0]- offset
        (pyramid_list2[0][:,:,0])[pyramid_list2[0][:,:,0]<0] = 0
        (pyramid_list2[1][:,:,0])[pyramid_list2[1][:,:,0]<0] = 0
        map_list2 = make_map(pyramid_list2)
        #edge for plus and sub
        pyramid_list_base, map_list_base = remap_image(pyramid_list_base, pyramid_list2, map_list_base, map_list2)
        min_base = remap_min_image(min_base, min2)
    plot_img = merge_all(min_base, pyramid_list_base)
    #plt.imshow(plot_img[:,:,0].astype(np.uint8), cmap="gray")
    #plt.show()
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_YUV2BGR)
    cv2.imwrite("result.jpg",plot_img)

if __name__=="__main__":
    main()
