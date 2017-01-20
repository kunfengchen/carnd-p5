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


# load the scaler
scaler_file_name = "x_scaler.pkl"
scaler_load = joblib.load(scaler_file_name)

# load the classifier model
model_file_name = "linearSVC_model.pkl"
svc_load = joblib.load(model_file_name)


def detect_vehicles_image_name(img_name):
    img = mpimg.imread(img_name)
    return detect_vehicles_image(img)


def detect_vehicles_image(img):
    pass


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
        #default="test_images/test6.jpg",
        #default="v1frames/frame1050.jpg",
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
