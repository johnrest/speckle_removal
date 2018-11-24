# File for general purposes, i.e. tools needed throught the module

from speck_rem import *


def get_list_images(directory, mask):
    "List of files containing the mask pattern within directory"
    lst = glob.glob(os.path.join(directory, mask))
    if not lst:
        print("Zero images found")
    return lst


def display_image(array, scale=1, title="Image"):
    """ Display and image with a scale factor and a figure title"""

    image = array_to_image(array)
    height, width = image.shape
    height = int(height*scale)
    width = int(width * scale)
    resampled = cv2.resize(image, (width, height))
    cv2.imshow(title, resampled)


def array_to_image(array):
    """ Transform and array to a 8bit image object """
    # TODO: include behavior to respond to flat images
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
        fname = os.path.join(target_folder, image_name_mask+"_{:03d}".format(count)+".tiff")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(fname, image)  # save frame as TIFF file
        success, image = video_capture.read()
        count += 1

    #Hard fix for removing extra frame fron trigger base acquisition.
    #Must delete last image since it repeats the first frame
    os.remove(fname)


def select_roi(array, wName="Select rectangle"):
    """ Use a cv2 tool to select a ROI from a numpy array"""
    image = array_to_image(array)
    cv2.namedWindow(wName, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(img=image, windowName=wName, fromCenter=False)
    cv2.destroyWindow(wName)
    return roi


def compute_pattern_batch(scale=4, batch_length=4*4/2):
    """Compute the set of small patterns that are grown into full sized dmd patterns"""
    U, V = np.meshgrid(range(0,scale), range(0,scale))
    U = U.flatten()
    V = V.flatten()
    off_positions = np.stack((U, V))
    off_positions = np.transpose(off_positions)
    np.random.shuffle(off_positions)
    off_positions = np.transpose(off_positions)

    batch = []
    off_positions_per_image = int(scale*scale/batch_length)          # Number of black pixels on each image

    for sel in (off_positions[:, i:i+off_positions_per_image] for i in range(0, off_positions.shape[1] - off_positions_per_image + 1, off_positions_per_image)):
        pattern = np.ones((scale, scale))
        pattern[sel[0,:], sel[1,:]] = 0         # off pixels with FC rule
        batch.append(pattern)

    return batch

def crop_image(filename_in, filename_out):
    """Centered crop an image to half their size"""
    image = p_Image.open(filename_in)
    array = np.array(image)

    h, w = array.shape

    array = array[ int(h/4):int(3*h/4), int(w/4):int(3*w/4) ]

    image = array_to_image(array)

    cv2.imwrite(filename_out, image, [cv2.IMWRITE_PNG_BILEVEL, 1])

def create_folder(parent_folder, name):
    new_folder = os.path.join(parent_folder, name)
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

