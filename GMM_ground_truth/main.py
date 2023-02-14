#Generating the GMM segmentation of the ice core images, along with cropping the outer tube from the images.
# the excel results calculate the density and amount of pixels, after finishing this part, move the .csv files to the result folder and run that code to get the final result. 
import cv2 as cv
import numpy as np
import glob
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
from multiprocessing import Process
import time
import math

def cropping_mask(gray):
# cropping the outer circle of the ice core when the circle is full and intact 
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    edges = cv.Canny(gray, 100, 200)
    edges = cv.dilate(edges, kernel=None, iterations=8)
    #edges = cv.dilate(edges, kernel=None, iterations=6)
    edges = cv.medianBlur(edges, 5)
    cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mask1 = np.zeros(gray.shape, np.uint8)
    cv.fillPoly(mask1, cnts, 1, )
    mask1 = cv.erode(mask1, kernel=None, iterations=18)
    edges[mask1 == 0] = 0
    dilated_edges = cv.dilate(edges, kernel=None, iterations=8)
    cnts, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mask2 = np.zeros(gray.shape, np.uint8)
    cv.fillPoly(mask2, cnts, 1, )
    mask2 = cv.medianBlur(mask2, 31)
    #mask2 = cv.erode(mask2, kernel=None, iterations=16)
    mask2 = cv.erode(mask2, kernel=None, iterations=16)
    cnts, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mask3 = np.zeros(gray.shape, np.uint8)
    # cv.drawContours(img,cnts,-1, color = (200,0,0), thickness = 4)
    cv.fillPoly(mask3, cnts, 1, )
    return mask3

def croping_mask_loos(gray):
# cropping the outer circle in case the circle is removed by ImageJ previously
    mask = np.zeros(gray.shape, np.uint8)
    cnts, _ = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = []
    for edge in cnts:
        for point in edge:
            for p in point:
                points.append(p)
    #hull = cv.convexHull(points)
    points = np.array(points)
    (x, y), radius = cv.minEnclosingCircle(points)
    center = (int(x), int(y))
    radius = int(radius)
    cv.circle(mask, center, radius, color =10, thickness = -1)
    return mask

def calculate_density (coreID, image_list):
    density_list = []
    ice_pixel_list = []

    for i in range (len(image_list[:])):
        path = image_list[i]
        #print(path)
        gray_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        original_shape = gray_image.shape
        #fro 30 Micron
        crop_mask = cropping_mask(gray_image)
        #crop_mask = croping_mask_loos(gray_image)
        #cv.imwrite(path[-10:-4]+'.png',crop_mask*100)



        gray_image[crop_mask == 0] = 0
        t2 = 60
        t1 = 95
        for j in range(gray_image.max()):
            if j< t1 and j>t2 :
                gray_image[gray_image == j] = 40

        gray_image = cv.GaussianBlur(gray_image, (5, 5), 0)


        gray_image = cv.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
                                  dtype=cv.CV_32F)
        # using the cropped values to make the GMM model
        img1_ = gray_image[crop_mask > 0].reshape((-1, 1))

        gmm_model = GMM(n_components=2, random_state=20, covariance_type='tied', init_params='kmeans').fit(img1_)
        #gray_image = cv.fastNlMeansDenoising(gray_image)

        ice_class = gmm_model.means_.round(3).tolist()
        ice_class_index = ice_class.index(max(ice_class))

        # giving whole image values for prediction for saving the shape of the array
        img2_ = gray_image.reshape((-1, 1))
         #-1 reshape means, in this case MxN
        gmm_labels = gmm_model.predict(img2_)

        area = len(gray_image[crop_mask > 0])
        #gmm_labels = gmm_model.predict(img2_)
        #print('shape = ', gmm_labels.shape)
        segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
        #segmented = gmm_labels.reshape(original_shape[0], original_shape[1],2)[:,:,ice_class_index]
        ice = segmented[segmented == ice_class_index]
        binary_img = np.zeros_like(segmented)
        binary_img[segmented == ice_class_index] = 200
        cv.imwrite('.//core_2_GMM_output//gmm_core{}_img{}.png'.format(coreID, i), binary_img)
        #Ice thresholding
        ice = len(ice)
        density = ice/area
    
        print ('core {} _ Density image_ {} = '.format(coreID, i), density)
        density_list.append(density)
        ice_pixel_list.append(ice)

        if i % 50 == 0:
            result =pd.DataFrame(zip(density_list, ice_pixel_list),columns =['density', 'ice_pixels'])
            result.to_csv('result_{}.csv'.format(coreID)) 
        elif i == len(image_list[:])-1:
            result =pd.DataFrame(zip(density_list, ice_pixel_list),columns =['density', 'ice_pixels'])
            result.to_csv('result_{}.csv'.format(coreID)) 

    print ('average density = ', sum(density_list)/ len(density_list))


if __name__ == '__main__':
    t1 = time.time()
    path1 = 'D://Faramarz_data//core_1//AA1_Canvas4096_FB_calibsnow30mueKF540_410btophx_8f//png'
    path2 = 'D://Faramarz_data//core_2//AA2_Canvas4096_FB_ex5_1_bag102bot_30mue_hx_8f//png'
    path3 = 'D://Faramarz_data//core_3//Canvas4096_FB_TED_bag43bot_30mue_hx_8f//png2'

    image_list = glob.glob(path2 + '//**.png')
    #image_list = image_list[800:801]

    print (' Number of images = ', len(image_list))
    #print(image_list)

    N_processor = 6
    step = int(len(image_list)//N_processor)


    #plt.imshow(gray_image)
    processor_list =[]
    for j in range(N_processor):
        if j == N_processor - 1:
            _img_list = image_list[j * step:]
            p = Process(target=calculate_density, args=(j,_img_list))
            processor_list.append(p)
            p.start()

        _img_list = image_list[j*step:(j+1)*step]
        p =  Process(target=calculate_density, args=(j,_img_list))
        processor_list.append(p)
        p.start()

    for p in processor_list:
        p.join()

    t2 = time.time()
    print ('Time = ', round(t2 - t1, 3))


