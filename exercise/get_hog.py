# Ported from class material

import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import scipy.misc
from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles


# Plot the examples
def show_hog(image, hog_images, titles):
    fig = plt.figure()
    n_hogs = len(hog_images)
    nx = math.ceil(math.sqrt(n_hogs+1))
    plt.subplot(nx, nx, 1)
    plt.imshow(image, cmap='gray')
    plt.title(titles[0])

    for i, hog_img in enumerate(hog_images, start=1):
        plt.subplot(nx, nx, i+1)
        plt.imshow(hog_img, cmap='gray')
        plt.title(titles[i])
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.show()


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=False, feature_vector=feature_vec)
        return features


def get_hog():
    images = glob.glob('data/saved_car/*.png')
    #images = glob.glob('data/saved_non_car/*.png')
    read_images = []
    hog_imgs = []

    for image in images:
            read_images.append(image)

    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(read_images))
    # Read in the image
    image = scipy.misc.imread(read_images[ind])

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    #orient = 9
    #pix_per_cell = 8
    #cell_per_block = 2
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False)
    hog_imgs.append(hog_image)
    features, hog_image = get_hog_features(image[:, :, 0], orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False)
    hog_imgs.append(hog_image)
    colors = (cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2HSV,
              cv2.COLOR_RGB2YUV, cv2.COLOR_RGB2YCrCb)
    titles = ['Car img', 'Gray', 'RGB-0', \
              'HLS-0', 'HLS-1', 'HLS-2', \
              'HSV-0', 'HSV-1', 'HSV-2', \
              'YUV-0', 'YUV-1', 'YUV-2', \
              'YCrCb-0', 'YCrCb-1', 'YCrCb-2']
    for c in colors:
        cimg = cv2.cvtColor(image, c)

        for i in (0, 1, 2):
            features, hog_image = get_hog_features(cimg[:,:,i], orient,
                                                   pix_per_cell, cell_per_block,
                                                   vis=True, feature_vec=False)
            hog_imgs.append(hog_image)

    show_hog(image, hog_imgs, titles)


# Main function
if __name__ == '__main__':
    get_hog()
