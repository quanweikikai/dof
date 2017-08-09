#! /usr/bin/python

import numpy as np
import cv2
import glob
from multiprocessing import Process
import matplotlib.pyplot as plt

size_list = [(6000,4000),(2400,1600),(1200,800), (810,540), (480,320)]
theta_ = [1,0.8,0.5,0.2,0.0]

def get_mask(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.int16)
    img = cv2.resize(img,(300,200))
    kernel = np.matrix([1,4,6,4,1])
    kernel = kernel.T*kernel/float(np.sum(kernel)**2)
    kernel2 = np.matrix([1,1,1,1,1])
    kernel2 = kernel2.T*kernel2/float(np.sum(kernel2)**2)
    blur = cv2.filter2D(img, -1 ,kernel)
    diff = np.abs(img-blur)
    diff = cv2.filter2D(diff, -1 ,kernel2)
    count = np.zeros(img.shape)
    idx = diff>10
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            top = i-20
            left = j-20
            if top < 0:
                top = 0
            if left < 0:
                left = 0
            bot = i+20
            right = j+20
            if bot > img.shape[0]:
                bot = img.shape[0]
            if right > img.shape[1]:
                right = img.shape[1]
            point_num = (bot-top)*(right-left)
            num_thresh = int(point_num*0.002)
            cc = idx[top:bot,left:right].ravel().tolist().count(True)
            if cc > num_thresh:
                count[i,j] = 255
    count = cv2.resize(count, (36,24))
    return count

def make_pyramid(image, mask, threshold=[0,0,0,0,0,0]):
    pyramid_list = []
    image = image.astype(np.int16)
    for i in xrange(len(size_list)-1):
        img_size = size_list[i]
        sub_size = size_list[i+1]
        kernel = np.matrix([1,4,6,4,1])
        kernel = kernel.T*kernel/256.
        kernel3 = np.matrix([1,1,1,1,1])
        kernel3 = kernel3*kernel3.T/25.
        #image_blr = cv2.filter2D(image,-1,kernel)
        image_sub = cv2.resize(image, sub_size)
        image_enlarge = cv2.resize(image_sub, img_size)
        #image_enlarge = cv2.filter2D(image_enlarge,-1,kernel)
        img_plus = image - image_enlarge
        img_minus = image_enlarge - image
        img_plus[img_plus<0] = 0
        img_minus[img_minus<0] = 0
        #plt.imsave("before%d.jpg"%i,(img_minus+img_plus).astype(np.uint8))
        # to use mask, first resize mask to proper size
        mask = cv2.resize(mask, size_list[i])
        # get mask under thresh, 100 as default
        idx_mask = mask<100
        img_plus[idx_mask] = 0
        img_minus[idx_mask] = 0
        mask[mask<100] = 0
        mask[mask>=100] = 255
        #plt.subplot(211)
        #plt.imshow(mask,cmap="gray")
        #plt.subplot(212)
        #plt.imshow((img_plus+img_minus).astype(np.uint8))
        #plt.show()
        #plt.imsave("after%d.jpg"%i,(img_minus+img_plus).astype(np.uint8))
        
        pyramid_list.append(img_plus)
        pyramid_list.append(img_minus)
        #if i==1:
        #    cv2.imwrite("edge_%d.jpg"%(2*i+1),img_minus.astype(np.uint8))
        #    cv2.imwrite("edge_%d.jpg"%(2*i),img_plus.astype(np.uint8))
        image = image_sub
    image = image.astype(np.uint8)
    #cv2.imwrite("edge_8.jpg",image)

    return pyramid_list,image

def make_map(pyramid_list, theta=0.):
    map_list = []
    kernel = np.matrix([1,4,6,4,1])
    kernel = kernel.T*kernel/256.
    kernel2 = np.matrix([[0,0,0,0,0],[0,-1,-1,-1,0],[0,-1,8,-1,0],[0,-1,-1,-1,0],[0,0,0,0,0]])
    kernel3 = np.matrix([1,1,1,1,1])
    kernel3 = kernel3*kernel3.T/25.
    for i in xrange(len(pyramid_list)/2):
        map_merged = (pyramid_list[2*i] + pyramid_list[2*i+1])
        #map_merged = cv2.filter2D(map_merged, -1, kernel*rate+kernel3*(1-rate))
        map_merged[map_merged<0] = 0
        map_merged[map_merged>255] = 255
        map_list.append(cv2.cvtColor(map_merged.astype(np.uint8),cv2.COLOR_BGR2GRAY))
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

def calc_deviation(image,rows,cols):
    assert rows%2 == 1
    assert cols%2 == 1
    if len(image.shape)==3:
        image = np.dot(image.astype(np.uint8),np.array([0.114,0.587,0.299]).T)
    dev_arr = np.zeros(image.shape)
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            top = i-(rows-1)/2
            left = j-(cols-1)/2
            if top < 0:
                top = 0
            if left < 0:
                left = 0
            bot = i+(rows-1)/2
            right = j+(cols-1)/2
            if bot > image.shape[0]:
                bot = image.shape[0]
            if right > image.shape[1]:
                right = image.shape[1]
            dev_arr[i,j] = np.std(image[top:bot,left:right])
    return dev_arr

def calc_pool(image, rows, cols):
    assert rows%2 == 1
    assert cols%2 == 1
    if len(image.shape)==3:
        image = np.dot(image.astype(np.uint8),np.array([0.114,0.587,0.299]).T)
    pool_arr = np.zeros(image.shape, dtype=np.int16)
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            top = i-(rows-1)/2
            left = j-(cols-1)/2
            if top < 0:
                top = 0
            if left < 0:
                left = 0
            bot = i+(rows-1)/2
            right = j+(cols-1)/2
            if bot > image.shape[0]:
                bot = image.shape[0]
            if right > image.shape[1]:
                right = image.shape[1]
            arr_sorted = np.sort(image[top:bot,left:right], axis=None)
            if arr_sorted.shape[0]<3:
                pool_arr[i,j] = float(np.sum(arr_sorted))/arr_sorted.shape[0]*3
            else:
                pool_arr[i,j] = np.sum(arr_sorted[-3:])
    return pool_arr


def remap_min_image(min1,min2,tag_img,threshold=[0,0,0,0,0],theta=0.0,index=1):
    assert min1.shape == min2.shape
    kernel = np.matrix([3,3,4,3,3])
    kernel = kernel.T*kernel/256.
    min1 = min1.astype(np.int16)
    min2 = min2.astype(np.int16)
    min_blur1 = cv2.filter2D(min1, -1, kernel)
    min_blur2 = cv2.filter2D(min2, -1, kernel)
    min_map1 = np.abs(min1-min_blur1)
    min_map2 = np.abs(min2-min_blur2)


    min_map1=np.dot(min_map1.astype(np.uint8),np.array([0.114,0.587,0.299]).T).astype(np.uint8)
    min_map2=np.dot(min_map2.astype(np.uint8),np.array([0.114,0.587,0.299]).T).astype(np.uint8)
    min_map1[min_map1<threshold[-1]] = 0
    min_map2[min_map2<threshold[-1]] = 0
    min_map1 = cv2.resize(min_map1,(120,80))
    min_map2 = cv2.resize(min_map2,(120,80))
    min_map1 = cv2.filter2D(min_map1, -1, kernel)
    min_map2 = cv2.filter2D(min_map2, -1, kernel)
    min_map1 = cv2.resize(min_map1,size_list[-1])
    min_map2 = cv2.resize(min_map2,size_list[-1])
    #plt.imshow(min_map2.astype(np.uint8),cmap="gray")
    #plt.show()
    #min_map1 = cv2.filter2D(min_map1, -1, kernel)
    #min_map2 = cv2.filter2D(min_map2, -1, kernel)
    #compare
    idx_1 = min_map1 > min_map2
    idx_2 = min_map1 <= min_map2
    remapped_img = np.zeros(min1.shape, dtype=np.uint8)
    remapped_img[idx_1] = min1[idx_1]
    remapped_img[idx_2] = min2[idx_2]
    tag_img[idx_2] = index
    remapped_img = remapped_img.astype(np.uint8)
    cv2.imwrite("remapped_min.jpg",remapped_img)
    return remapped_img, tag_img

    
def merge_all(min_img, pyramid_list):
    assert len(pyramid_list)%2 == 0
    merged_img = min_img.astype(np.int16)
    kernel = np.matrix([1,4,6,4,1])
    kernel = kernel.T*kernel/256.
    for i in xrange(len(pyramid_list)/2):
        merged_img = cv2.resize(merged_img, size_list[-2-i])
        #merged_img = cv2.filter2D(merged_img, -1, kernel)
        #pay attention: merge image inversely(from back to front)
        merged_img -= pyramid_list[-i*2-1]
        merged_img += pyramid_list[-i*2-2]
        merged_img[merged_img>255] = 255
        merged_img[merged_img<0] = 0
        save_img = merged_img.astype(np.uint8)
        cv2.imwrite("%d_middle.jpg"%i,save_img)
    merged_img = merged_img.astype(np.uint8)
    np.save("merge.npy", [min_img, pyramid_list])
    for i , py in enumerate(pyramid_list):
        #cv2.imwrite("edge_%d.jpg"%i, py.astype(np.uint8))
        pass
    return merged_img

image_folder = "./group2/"
import sys
if len(sys.argv) >1:
    image_folder = sys.argv[1]
def main(theta,threshold):
    image_name = glob.glob(image_folder+"*.JPG")
    image_name = sorted(image_name)
    print image_name
    image1 = cv2.imread(image_name[0])
    # get mask for image
    # mask size is 36*24
    mask_base = get_mask(image1)
    pyramid_list_base, min_base = make_pyramid(image1, mask_base, threshold)
    map_list_base = make_map(pyramid_list_base,theta)
    min_img_list = [min_base]
    tag_img = np.zeros(min_base.shape,dtype=np.uint8)
    for i in xrange(1,len(image_name)):
        print "process No.%d image"%(i)
        image2 = cv2.imread(image_name[i])
        assert image1.shape == image2.shape
        # get mask for image
        mask2 = get_mask(image2)
        pyramid_list2, min2 = make_pyramid(image2, mask2, threshold)
        map_list2 = make_map(pyramid_list2)
        #edge for plus and sub
        pyramid_list_base, map_list_base = remap_image(pyramid_list_base, pyramid_list2, map_list_base, map_list2)
        min_base, tag_img= remap_min_image(min_base, min2, tag_img, threshold, theta, i)
        min_img_list.append(min_base)
    plot_img = merge_all(min_base, pyramid_list_base)
    cv2.imwrite("result_%d_%d_%d_%d.jpg"%(threshold[2],threshold[3],threshold[4],threshold[5]),plot_img)
    np.save("min.npy",tag_img)
    np.save("min_list.npy",min_img_list)
    for j,img_ in enumerate(min_img_list):
        cv2.imwrite("min_%d.jpg"%j,img_)
        
        
if __name__=="__main__":
    main(0.0,[0,0,0,0,0,0])
    '''
    #for i in xrange(4):
            #for k in xrange(0,2):
                #main(0.2*i, [0,0,j,k])
                #for l in xrange(16,20,2):
                    #for j in xrange(12,15,2):
                        p1 = Process(target = main, args=(0.0, [0, 0, 0, 12, 16, 24]))
                        p2 = Process(target = main, args=(0.0, [0, 0, 0, 12, 16, 26]))
                        p3 = Process(target = main, args=(0.0, [0, 0, 0, 12, 16, 28]))
                        p4 = Process(target = main, args=(0.0, [0, 0, 0, 12, 16, 30]))
                        p1.start()
                        p2.start()
                        p3.start()
                        p4.start()
                        p1.join()
                        p2.join()
                        p3.join()
                        p4.join()
                        '''
