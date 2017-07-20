#! /usr/bin/python

import numpy as np
import cv2
import glob
#theta is laplacian weight with gaussian weight=(1-theta)

def make_pyramid(image, threshold=[0,0,2,5]):
    pyramid_list = []
    image = image.astype(np.int16)
    for i in xrange(4):
        img_size = (image.shape[1],image.shape[0])
        sub_size = (image.shape[1]/2,image.shape[0]/2)
        kernel = np.matrix([1,4,6,4,1])
        kernel = kernel.T*kernel/256.
        image_blr = cv2.filter2D(image,-1,kernel)
        image_sub = cv2.resize(image_blr, sub_size)
        image_enlarge = cv2.resize(image_sub, img_size)
        image_enlarge = cv2.filter2D(image_enlarge,-1,kernel)
        img_plus = image - image_enlarge
        img_minus = image_enlarge - image
        if (i>=2):
            img_plus[img_plus<threshold[i]] = 0
            img_minus[img_minus<threshold[i]] = 0
        img_plus[img_plus<0] = 0
        img_minus[img_minus<0] = 0
        pyramid_list.append(img_plus)
        pyramid_list.append(img_minus)
        image = image_sub
    image = image.astype(np.uint8)

    return pyramid_list,image

def make_map(pyramid_list, theta=0.2):
    map_list = []
    kernel = np.matrix([1,4,6,4,1])
    kernel = kernel.T*kernel/256.
    kernel2 = np.matrix([[0,0,0,0,0],[0,-1,-1,-1,0],[0,-1,8,-1,0],[0,-1,-1,-1,0],[0,0,0,0,0]])
    for i in xrange(len(pyramid_list)/2):
        map_merged = (pyramid_list[2*i] + pyramid_list[2*i+1])
        map_merged = cv2.filter2D(map_merged, -1, (1-theta)*kernel+kernel2*theta)
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
        img_plus = np.zeros(pyramid_list1[i*2].shape, dtype=np.uint8)
        img_minus = np.zeros(pyramid_list1[i*2+1].shape, dtype=np.uint8)
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
    kernel = np.matrix([1,1,1,1,1])
    kernel = kernel.T*kernel/25.
    min_blur1 = cv2.filter2D(min1, -1, kernel)
    min_blur2 = cv2.filter2D(min2, -1, kernel)
    min_map1 = (min1-min_blur1) + (min_blur1-min1)
    min_map2 = (min2-min_blur2) + (min_blur2-min2)
    min_map1 = cv2.filter2D(min_map1, -1, kernel)
    min_map2 = cv2.filter2D(min_map2, -1, kernel)
    #compare
    idx_1 = min_map1 > min_map2
    idx_2 = min_map1 <= min_map2
    remapped_img = np.zeros(min1.shape, dtype=np.uint8)
    remapped_img[idx_1] = min1[idx_1]
    remapped_img[idx_2] = min2[idx_2]
    return remapped_img

    
def merge_all(min_img, pyramid_list):
    assert len(pyramid_list)%2 == 0
    merged_img = min_img.astype(np.int16)
    kernel = np.matrix([1,4,6,4,1])
    kernel = kernel.T*kernel/256.
    for i in xrange(len(pyramid_list)/2):
        merged_img = cv2.resize(merged_img, (merged_img.shape[1]*2, merged_img.shape[0]*2))
        merged_img = cv2.filter2D(merged_img, -1, kernel)
        #pay attention: merge image inversely(from back to front)
        merged_img -= pyramid_list[-i*2-1]
        merged_img += pyramid_list[-i*2-2]
        #plot = merged_img.astype(np.uint8)
        #cv2.imshow("img",plot)
        #cv2.waitKey(0)
    merged_img[merged_img>255] = 255
    merged_img[merged_img<0] = 0
    merged_img = merged_img.astype(np.uint8)
    return merged_img

image_folder = "./image/"
def main(theta,threshold):
    image_name = glob.glob(image_folder+"*.JPG")
    image1 = cv2.imread(image_name[0])
    pyramid_list_base, min_base = make_pyramid(image1,threshold)
    map_list_base = make_map(pyramid_list_base,theta)
    for i in xrange(1,len(image_name)):
        print "process No.%d image"%(i)
        image2 = cv2.imread(image_name[i])
        assert image1.shape == image2.shape
        pyramid_list2, min2 = make_pyramid(image2)
        map_list2 = make_map(pyramid_list2)
        #edge for plus and sub
        pyramid_list_base, map_list_base = remap_image(pyramid_list_base, pyramid_list2, map_list_base, map_list2)
        min_base = remap_min_image(min_base, min2)
    plot_img = merge_all(min_base, pyramid_list_base)
    cv2.imwrite("result_%d_%1d_%1f.jpg"%(threshold[2],threshold[3],theta),plot_img)
        
        


if __name__=="__main__":
    for i in xrange(4):
        for j in xrange(0,5,1):
            for k in xrange(0,15,2):
                main(0.2*i, [0,0,j,k])
    
