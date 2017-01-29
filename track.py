
import numpy as np
from pyimagesearch.nms import non_max_suppression_slow


class Tracker:
    def __init__(self):
        """
        Init
        """
        frames = []


    def track(self, windows):
        """
        Track the bounding windows per frame
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
