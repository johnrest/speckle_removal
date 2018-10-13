# File for general purposes, i.e. tools needed throught the module

from speck_rem import *


def get_list_images(directory, mask):
    "List of files containing the mask pattern within directory"
    return glob.glob(os.path.join(directory, mask))


def display_image(array, scale=1, title="Image"):
    """ Display and image with a scale factor and a title"""

    image = array_to_image(array)
    height, width = image.shape
    height = int(height*scale)
    width = int(width * scale)
    resampled = cv2.resize(image, (width, height))
    cv2.imshow(title, resampled)


def array_to_image(array):
    """ Transform and array to a 8bit image object """

    array = array.astype(np.float64)
    array -= array.min()
    array *= 255 / array.max()
    image = array.astype(np.uint8)
    return image


def extract_frames_from_video(target_folder, video_filename, image_name_mask):
    """ Extract all frames from a video and store as png files in the same folder"""
    video_capture = cv2.VideoCapture(os.path.join(target_folder, video_filename))
    success, image = video_capture.read()
    count = 0
    while success:
        fname = os.path.join(target_folder, image_name_mask+"_{:03d}".format(count)+".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(fname, image)  # save frame as PNG file
        success, image = video_capture.read()
        count += 1

    #Hard fix for removing extra frame fron trigger base acquisition.
    #Must delete last image since it repeats the first frame
    os.remove(fname)

    #TODO: imporve this hard fix


def select_roi(array, wName="Select rectangle"):
    """ Use a cv2 tool to select a ROI from a numpy array"""
    image = array_to_image(array)
    cv2.namedWindow(wName, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(img=image, windowName=wName, fromCenter=False)
    cv2.destroyWindow(wName)
    return roi
