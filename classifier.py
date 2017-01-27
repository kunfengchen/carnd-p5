# Ported from class material

import scipy.misc
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


CLASSIFIER_IMG_SIZE = (64, 64) # Image size for training and detecting
MODEL_FIEL_NAME = "linearSVC_model.pkl"
SCALER_FILE_NAME = "x_scaler.pkl"

PIX_PER_CELL= 8  # Default pixels per cell for hog function
CELLS_PER_BLOCK = 2  # Default cells per block for hog function
ORIENTS_HOG = 8  # Default orientations for hog function
HOG_CHANNEL = 0  # Default color channel for hog function
CSPACE = 'YCrCb'  # Default color space for hog function
#CSPACE = 'RGB'  # Default color space for hog function
DECISION_THRESHOLD = 4  # Default positive threshold for decision_function


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_one_features(image, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
         # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Call get_hog_features() with vis=False, feature_vec=True
    hog_features = get_hog_features(
        feature_image[:,:,hog_channel], orient,
        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_features_2 = get_hog_features(  # Add addtional hog for channel 2
             feature_image[:,:,2], orient,
             pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_features_3 = get_hog_features(  # Add addtional hog for channel 2
        feature_image[:,:,1], orient,
        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    #return np.concatenate((spatial_features, hist_features, hog_features))
    return np.concatenate((hist_features, hog_features, hog_features_2, hog_features_3))
    #return np.concatenate((spatial_features, hog_features, hog_features_2, hog_features_3))
    #return np.concatenate(( hist_features, hog_features))
    #return hog_features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = scipy.misc.imread(file)
        # Append the new feature vector to the features list
        one_features = extract_one_features(
           image, cspace=cspace, spatial_size=spatial_size,
           hist_bins=hist_bins, hist_range=hist_range, orient=orient,
           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
           hog_channel=hog_channel)
        features.append(one_features)
    # Return list of feature vectors
    return features


def get_image_names(glob_string):
    image_glob_names = glob.glob(glob_string)
    image_names = []
    for image in image_glob_names:
        image_names.append(image)
    return image_names


def train():
    car_image_glob_string = 'data/vehicles/*/*.png'
    cars = get_image_names(car_image_glob_string)
    notcars_image_glob_string = 'data/non-vehicles/*/*.png'
    notcars = get_image_names(notcars_image_glob_string)

    t=time.time()
    print("Extrating features", len(cars), " car images ",
          len(notcars), " non car images")
    car_features = extract_features(cars, cspace=CSPACE, spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256),
                                    orient=ORIENTS_HOG, pix_per_cell=PIX_PER_CELL,
                                    cell_per_block=CELLS_PER_BLOCK,
                                    hog_channel=HOG_CHANNEL)
    notcar_features = extract_features(notcars, cspace=CSPACE, spatial_size=(32, 32),
                                       hist_bins=32, hist_range=(0, 256),
                                       orient=ORIENTS_HOG, pix_per_cell=PIX_PER_CELL,
                                       cell_per_block=CELLS_PER_BLOCK,
                                       hog_channel=HOG_CHANNEL)
    t2 = time.time()
    print(t2-t, 'seconds to load imags extract features')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    print("X shape = ", X.shape)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Save the scaler
    joblib.dump(X_scaler, SCALER_FILE_NAME)
    print("X_scaler saved to:", SCALER_FILE_NAME)

    # Reload the scaler
    scaler_load = joblib.load(SCALER_FILE_NAME)

    # Apply the scaler to X
    #scaled_X = X_scaler.transform(X)
    scaled_X = scaler_load.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    print("Training LinearSVC")
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(t2-t, 'seconds to train SVC...')
    # Check the score of the SVC
    print('Train Accuracy of SVC = ', svc.score(X_train, y_train))
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

    # Save the model
    joblib.dump(svc, MODEL_FIEL_NAME)
    print("classified model saved to:", MODEL_FIEL_NAME)

    # Reload the model
    svc_load = joblib.load(MODEL_FIEL_NAME)

    # Check the score of the SVC for reloading
    print('Train Accuracy of SVC after reload = ', svc_load.score(X_train, y_train))
    print('Test Accuracy of SVC after reload = ', svc_load.score(X_test, y_test))


    # Check the prediction time for a single sample
    t=time.time()
    prediction = svc_load.predict(X_test[0].reshape(1, -1))
    t2 = time.time()
    print(t2-t, 'Seconds to predict with SVC')


if __name__ == '__main__':
    train()
