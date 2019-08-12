# File for general purposes, i.e. tools needed throught the module

from speck_rem import *
# from .holography import Image

def get_list_images(directory, mask):
    """
    List of files containing the mask pattern within the directory
    :param directory: string with full directory
    :param mask: string with mask to query for names
    :return: list of strings with file names
    """
    lst = glob.glob(os.path.join(directory, mask))
    if not lst:
        print("Zero images found")
    return lst


def display_image(array, scale=1, title="Image"):
    """
    Display and image with a scale resize factor and a figure title. Based on opencv imshow.
    :param array: numpy array
    :param scale: float as scaling factor
    :param title: string for the window title
    :return: None
    """
    image = array_to_image(array)
    height, width = image.shape
    height = int(height*scale)
    width = int(width * scale)
    resampled = cv2.resize(image, (width, height))
    cv2.imshow(title, resampled)


def array_to_image(array):
    """
    Transform and array to a 8bit image object
    :param array: numpy array
    :return: 8 bit image
    """

    array = array.astype(np.float64)
    if array.min() != array.max():
        array -= array.min()
        array *= 255 / array.max()

    image = array.astype(np.uint8)
    return image


def extract_frames_from_video(target_folder, video_filename, image_name_mask):
    """
    Extract all frames from a video and store as tiff files in the same folder
    :param target_folder: string with the full folder name containing the video
    :param video_filename: string with the video file name
    :param image_name_mask: string with the mask for naming the list of extracted frames
    :return: None
    """
    video_capture = cv2.VideoCapture(os.path.join(target_folder, video_filename))
    success, image = video_capture.read()
    count = 0
    while success:
        filename = os.path.join(target_folder, image_name_mask+"_{:03d}".format(count)+".tiff")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filename, image)
        success, image = video_capture.read()
        count += 1

    # WARNING: Hard fix for removing extra frame fron trigger base acquisition.
    # Must delete last image since it repeats the first frame
    os.remove(filename)


def select_roi(array, window_name="Select rectangle"):
    """
    Use a cv2 tool to select a ROI from a numpy array
    :param array: numpy array
    :param window_name: string with a user defined window name
    :return: list with roi as (a,b,w,h)
    """
    image = array_to_image(array)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(img=image, windowName=window_name, fromCenter=False)
    cv2.destroyWindow(window_name)
    return roi


def compute_pattern_batch(scale=4, batch_length=4*4/2):
    """
    Compute the set of small patterns that are grown into full sized dmd patterns
    :param scale:
    :param batch_length:
    :return:
    """
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
    """
    Centered crop an image read from file, to half its size.
    :param filename_in: string with the full file name
    :param filename_out: string with the full file name for the output image
    :return: None
    """
    image = cv2.imread(filename_in, cv2.IMREAD_GRAYSCALE)
    array = np.array(image)

    h, w = array.shape

    array = array[ int(h/4):int(3*h/4), int(w/4):int(3*w/4) ]

    image = array_to_image(array)

    cv2.imwrite(filename_out, image, [cv2.IMWRITE_PNG_BILEVEL, 1])


def create_folder(parent_folder, name):
    """
    Create a new directory inside an specified location. It does nothing if it already exists
    :param parent_folder: string with the parent folder
    :param name: string the new directory name
    :return: string with the new folder name
    """
    new_folder = os.path.join(parent_folder, name)
    os.makedirs(new_folder, exist_ok=True)
    return new_folder


def image_profile(files, pts):
    """
    Select a profile from a gray scale image, based on two clicks
    :param files: List of string with full file names
    :param pts: Numpy Array containing the points
    :return: profile_data, numpy array with data in columns
    """

    # Check two points are available
    assert pts.shape[0] == 2, "Points are not correct"
    assert pts.shape[0] >= 2, "Points are not correct"

    if pts.shape[1] > 2:
        pts = pts[:, 0:2]           # Discard unnecessary points

    # Use the clicked points to select lines for all files and store
    # inside a numpy array as column vectors

    data = []
    for itr, file in enumerate(files):
        img = Image()
        img.read_image_file_into_array(file)
        print("Processing image:", itr)

        data.append(bresenham_march(img.image_array, pts[:, 0], pts[:, 1]))

    data = np.array(data)

    return data


def bresenham_march(img, p1, p2):
    """
    Shameless copy from https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
    :param img: Image array
    :param p1: First point
    :param p2: Second point
    :return: ret: list containing the pixel data
    """
    x1 = p1[1]
    y1 = p1[0]
    x2 = p2[1]
    y2 = p2[0]

    print("P1", p1)
    print("P2", p2)

    # tests if any coordinate is outside the image
    if (
            x1 >= img.shape[0]
            or x2 >= img.shape[0]
            or y1 >= img.shape[1]
            or y2 >= img.shape[1]
    ):  # tests if line is in image, necessary because some part of the line must be inside,
        # it respects the case that the two points are outside
        if not cv2.clipLine((0, 0, *img.shape), p1, p2):
            print("not in region")
            return

    steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # takes left to right
    also_steep = x1 > x2
    if also_steep:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = math.fabs(y2 - y1)
    error = 0.0
    delta_error = 0.0
    # Default if dx is zero
    if dx != 0:
        delta_error = math.fabs(dy / dx)

    y_step = 1 if y1 < y2 else -1

    y = y1
    ret = []
    for x in range(x1, x2):
        p = (y, x) if steep else (x, y)
        if p[0] < img.shape[0] and p[1] < img.shape[1]:
            # ret.append((p, img[p]))
            ret.append(img[p])              # John: only store the gray values
        error += delta_error
        if error >= 0.5:
            y += y_step
            error -= 1
    if also_steep:  # because we took the left to right instead
        ret.reverse()
    return ret


