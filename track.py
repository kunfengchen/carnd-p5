from collections import deque
import numpy as np
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import cv2

from pyimagesearch.nms import non_max_suppression_slow

IMG_WIDTH=1280
IMG_HEIGTH=720
TRACK_N_FRAMES = 5
HEATMAP_THRESHOLD = 8

class Tracker:

    def __init__(self,
                 track_frame_number=TRACK_N_FRAMES,
                 heatmap_threshold=HEATMAP_THRESHOLD):
        """
        Frame tracking class init.
        :param track_frame_number: Speicify how many frame to track
        :param heatmap_threshold: Speicify threshold for the heatmap
        """
        self.heatmap_threshold=heatmap_threshold
        self.track_frame_number=track_frame_number
        self.frames = deque(maxlen=track_frame_number)


    def track_nms(self, windows, view=False):
        """
        Track the bounding windows per frame using non maximum suppression
        No frame tracking yet
        :param windows: the bounding windows
        :return: the tracked objects
        """
        window_list = []
        for w in windows:
            (x1, y1), (x2, y2) = w
            window_list.append([x1, y1, x2, y2])
        picks = non_max_suppression_slow(np.array(window_list), 0.005)
        pick_windows = []
        for p in picks:
            [x1, y1, x2, y2] = p
            pick_windows.append(((x1, y1), (x2, y2)))
        return pick_windows


    def track_labels(self, windows, view=False):
        """
        Track the bounding windows using heat map and measurements.label
        :param windows: the bounding windows
        :param view: to show the progression view or not
        :return: the labels
        """
        self.frames.append(windows)
        heatmap = np.zeros((IMG_HEIGTH, IMG_WIDTH), dtype=np.uint8)
        tracking_windows = []
        for frame in self.frames:
            tracking_windows.extend(frame)

        Tracker.add_heat(heatmap, tracking_windows)
        heatmap_thres = Tracker.apply_threshold(heatmap, self.heatmap_threshold)
        labels = Tracker.label_cars(heatmap_thres)
        if view:
            Tracker.show_image(heatmap, title='heatmap', cmap=plt.get_cmap('jet'))
            Tracker.show_image(heatmap_thres, title='heatmap threshold', cmap=plt.get_cmap('jet'))
            Tracker.show_image(labels[0], title='labels', cmap=plt.get_cmap('gray'))
            #Tracker.show_image(track_img, title='trakced img', cmap=plt.get_cmap('gray'))
        #return track_img
        return labels

    ### source from class material
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    ### source from class material
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below ehe threshold
        heatmap_thres = np.copy(heatmap)
        heatmap_thres[heatmap_thres <= threshold] = 0
        # Return thresholded map
        return heatmap_thres

    ### source from class material
    def label_cars(heatmap):
        labels = label(heatmap)
        return labels


    ### source from class material
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
        # Return the image
        return img


    def show_image(img, title=None, cmap=None ):
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.show()

