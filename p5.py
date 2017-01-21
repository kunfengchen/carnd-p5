import argparse
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from exercise.sliding_window import \
    slide_window, draw_boxes, show_grid_view


def load_scaler():
    # load the scaler
    scaler_file_name = "x_scaler.pkl"
    scaler_load = joblib.load(scaler_file_name)
    return scaler_load


def load_classifier_model():
    # load the classifier model
    model_file_name = "linearSVC_model.pkl"
    svc_load = joblib.load(model_file_name)
    return svc_load


def predit_vehicles(img, windows,
                   scaler = load_scaler(),
                   classifier = load_classifier()):
    """
    Predict if image in sliding windows is a car
    :param img: the image to look for
    :param windows: list of sub windows
    :param scaler: scale the sub windows image features
    :param classifier: classify a car or not
    :return: list of sub windows classified as car
    """
    car_feature_list = []
    return windows[classifier.predit(car_feature_list)]


def detect_vehicles_image_name(img_name):
    img = mpimg.imread(img_name)
    return detect_vehicles_image(img)


def detect_vehicles_image(input_img,
                          scaler=load_scaler(),
                          classifier=load_classifier_model(),
                          view=True):
    img = np.copy(input_img)
    # diffent sliding window sizes
    xy_windows=[(100, 100), (140, 140), (180, 180)]
    for xy_window in xy_windows:
        img_height = img.shape[0]
        img_width  = img.shape[1]
        view_height = int(img_height/2)
        y_start_stop=[view_height, img_height]
        xy_nums = (int(view_height/xy_window[1])*2,
                   int(img_width/xy_window[0])*2)

        window_list = slide_window(
            img, xy_window=xy_window, y_start_stop=y_start_stop)
        if view:
            show_grid_view(img, window_list, xy_nums=xy_nums)
        cars_window_list = predit_vehicles(
            img, window_list,
            scaler=scaler, classifier=classifier)

    return img


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
        default="test/frame981.jpg",
        help='image to be processed')
    parser.add_argument(
        '--video',
        default="project_video.mp4")
    parser.add_argument(
        '--visual',
        action='store_true',
        default=False)
    args = parser.parse_args()
    view = None
    if args.visual:
        view = None
    if args.train:
        pass
    if args.image is not None:
        detect_vehicles_image_name(args.image)
        exit()
    if args.video is not None:
        detect_vehicles_video(args.video)
