# Ported from class material

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc

SAVE_PATH = 'data/saved/'

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def get_sub_image(img, window):
    """
    Get a small set of the image specified by the window
    :param img: The whole image
    :param window: the sub image region
    :return: the sub image
    """
    (startx,starty), (endx, endy)= window
    return img[starty:endy, startx:endx, :]


def show_grid_view(img, bboxes, xy_nums=(8, 7)):
    """
    Plot each sub window images
    :param img: the whole image
    :param bboxes: the sub window box
    :param xy_nums: x,y size of grid display
    :return:
    """
    for i, box in enumerate(bboxes, start=1):
        (startx,starty), (endx, endy)= box
        plt.subplot(xy_nums[0], xy_nums[1], i, xticks=[], yticks=[])
        #plt.imshow(img[starty:endy, startx:endx, :])
        plt.imshow(get_sub_image(img, box))
    plt.show()


def save_sub_images(
        img, windows, size=(64, 64),
        path=SAVE_PATH, prefix='w', start_num=0):
    """
    Save images specified in windows to files
    :param img:
    :param windows:
    :param size: the final image size to be saved
    :return:
    """
    for i, window in enumerate(windows):
        fname = path + str(prefix) +str(i+start_num) + '.png'
        sub_img = get_sub_image(img, window)
        re_img = scipy.misc.imresize(sub_img, size)
        scipy.misc.imsave(fname, re_img)


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


if __name__ == '__main__':
    #image = mpimg.imread('bbox-example-image.jpg')
    image = scipy.misc.imread('../test/frame350.jpg')

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                           xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()
    #show_grid_view(image, windows, xy_nums=(10,19))
    show_grid_view(image, windows, xy_nums=(20,45))
