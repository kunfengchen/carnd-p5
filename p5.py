import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.misc
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from exercise.sliding_window import \
    slide_window, draw_boxes, show_grid_view, \
    get_sub_image, save_sub_images, SAVE_PATH
from classifier import extract_one_features, \
    train, CLASSIFIER_IMG_SIZE, MODEL_FIEL_NAME, SCALER_FILE_NAME, \
    PIX_PER_CELL, CELLs_PER_BLOCK, HOG_CHANNEL, ORIENTS_HOG


def load_scaler():
    # load the scaler
    scaler_load = joblib.load(SCALER_FILE_NAME)
    return scaler_load


def load_classifier_model():
    # load the classifier model
    svc_load = joblib.load(MODEL_FIEL_NAME)
    return svc_load

def show_features(features_list, xy_nums=(20, 20)):
     for i, features in enumerate(features_list, start=1):
        plt.subplot(xy_nums[0], xy_nums[1], i, xticks=[], yticks=[])
        plt.plot(features)
     plt.show()


def predit_vehicles(img, windows,
                   scaler = None,
                   classifier = None,
                   view=True,
                   save_path=None,
                   prefix=None):
    """
    Predict if image in sliding windows is a car
    :param img: the image to look for
    :param windows: list of sub windows
    :param scaler: scale the sub windows image features
    :param classifier: classify a car or not
    :param save_path: If specified, save detected images to the path
    :return: list of sub windows classified as car
    """
    car_features_list = []
    for window in windows:
        sub_img = cv2.resize(
            get_sub_image(img, window), CLASSIFIER_IMG_SIZE)
        one_features = extract_one_features(
            sub_img,
            cspace='RGB', spatial_size=(32, 32),
            hist_bins=32, hist_range=(0, 256),
            orient=ORIENTS_HOG, pix_per_cell=PIX_PER_CELL,
            cell_per_block=CELLs_PER_BLOCK,
            hog_channel=HOG_CHANNEL)
        car_features_list.append(
           # scaler.transform(np.asarray(one_features, dtype=np.float64)))
             one_features)
        scaled_list = scaler.fit_transform(car_features_list)
    #if view:
    #    show_features(scaled_list)
    predicts = classifier.predict(scaled_list)
    predicts_bool = predicts.astype(np.bool)
    car_windows = []
    for p, w in zip(predicts_bool, windows):
        if p:
            car_windows.append(w)
    #if view:
    #    show_grid_view(img, car_windows)
    if save_path is not None:
        save_sub_images(img, car_windows,
                        path=save_path,
                        prefix=prefix,
                        size=CLASSIFIER_IMG_SIZE)
    return car_windows


def detect_vehicles_image_name(img_name,
                               scaler=None,
                               classifier=None,
                               save_path=None,
                               prefix=None,
                               view=True):
    img = scipy.misc.imread(img_name)
    return detect_vehicles_image(img,
                                 scaler=scaler,
                                 save_path=save_path,
                                 prefix=None,
                                 classifier=classifier,
                                 view=view)


def detect_vehicles_image(input_img,
                          scaler=None,
                          classifier=None,
                          save_path=None,
                          prefix=None,
                          view=True):
    img = np.copy(input_img)
    # diffent sliding window sizes
    #xy_windows=[(80, 80), (100, 100), (140, 140), (180, 180)]
    xy_windows=[(180, 180)]
    xy_overlap=(0.50, 0.50)

    for i, xy_window in enumerate(xy_windows):
        img_height = img.shape[0]
        img_width  = img.shape[1]
        view_height = int(img_height/2)
        y_start_stop=[view_height, img_height]
        xy_nums = (int(view_height/xy_window[1])*2,
                   int(img_width/xy_window[0])*2)

        window_list = slide_window(img, xy_window=xy_window,
            xy_overlap=xy_overlap, y_start_stop=y_start_stop)
        prefix = 'w' + str(i)
        #if view:
            #show_grid_view(img, window_list, xy_nums=xy_nums)
        car_window_list = predit_vehicles(
            img, window_list,
            scaler=scaler, classifier=classifier,
            save_path=save_path,
            prefix=prefix)
        if view:
            window_img = draw_boxes(img, car_window_list, color=(0, 0, 255), thick=6)
            plt.imshow(window_img)
            plt.show()

    return window_img


def detect_vehicles_video(video_name):
    pass


## Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        help='Train the classifier')
    parser.add_argument(
        '--image',
        #default="test/frame981.jpg",
        default="test/frame350.jpg",
        help='image to be processed')
    parser.add_argument(
        '--v/ideo',
        default="project_video.mp4")
    parser.add_argument(
        '--visual',
        action='store_true',
        default=False)
    args = parser.parse_args()
    view = None
    if args.visual:
        view = True
    if args.train:
        train()
        exit()
    if args.image is not None:
        detect_vehicles_image_name(
            args.image,
            save_path=SAVE_PATH,
            classifier=load_classifier_model(),
            scaler=load_scaler())
            #scaler=StandardScaler())
        exit()
    if args.video is not None:
        detect_vehicles_video(args.video)
