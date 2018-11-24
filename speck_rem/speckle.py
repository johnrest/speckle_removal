# Tools related to speckle computations

from speck_rem import *

def speckle_correlation_coefficient(images_list, roi=True):
    """Compute the speckle correlation coeficient for a list of reconstructed images"""
    first_image = Image()
    first_image.read_image_file_into_array(images_list[0])

    if roi is True:                                                 # Select ROI to compute correlation in
        roi = select_roi(first_image.image_array, "Select a ROI to compute the speckle correlation")
    else:
        roi = first_image.image_array.shape                               # Compute correlation over full image

    speckle_corr_coeff = np.empty((len(images_list), len(images_list)), dtype=float)

    for ii, file_ii in enumerate(images_list):
        image_ii = Image()
        image_ii.read_image_file_into_array(file_ii)
        amp_ii = image_ii.image_array[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        for jj, file_jj in enumerate(images_list):
            image_jj = Image()
            image_jj.read_image_file_into_array(file_jj)
            amp_jj = image_jj.image_array[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            speckle_corr_coeff[ii, jj] = np.abs(np.sum((amp_ii-np.mean(amp_ii))*(amp_jj-np.mean(amp_jj))))\
                                     /np.sqrt(np.sum(np.power(amp_ii-np.mean(amp_ii), 2)) * np.sum(np.power(amp_jj-np.mean(amp_jj), 2)))
        print("...")

    return speckle_corr_coeff


def speckle_contrast(images_list, roi=None, mode=None):
    """ Compute the correlation coefficient for a list of reconstructed images,
        returns a tuple with the coefficient for average and standard deviation
        modes. Creates an image stack with a ROI selection to avoid memory issues """

    first_image = Image()
    first_image.read_image_file_into_array(images_list[0])

    if not roi:
        roi = select_roi(first_image.image_array, "Select a ROI to compute the speckle contrast")

    stack = np.empty((roi[3], roi[2], len(images_list)), dtype=float)

    sum_image = np.zeros((roi[3], roi[2]), dtype=float)
    average_coeff = []
    standard_dev_coeff = []

    for itr, item in enumerate(images_list):
        current = Image()
        current.read_image_file_into_array(item)

        stack[:, :, itr] = current.image_array[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]

        sum_image += stack[:, :, itr]

        standard_dev_image = np.std(stack, axis=2)

        average_coeff.append(np.std(sum_image/(itr+1))/np.average(sum_image/(itr+1)))
        standard_dev_coeff.append(np.std(standard_dev_image)/np.average(standard_dev_image))

    return average_coeff, standard_dev_coeff


def superposition_standard_dev(images_list, filename, format):
    """
    Compute the superposition image as the axial standard deviation of the stack pixel by pixel
    :param images_list: list of full filenames
    :param filename: full filename to write the final image
    """
    first_image = Image()
    first_image.read_image_file_into_array(images_list[0])

    stack = np.empty((*first_image.image_array.shape, len(images_list)), dtype=float)

    for itr, item in enumerate(images_list):
        current = Image()
        current.read_image_file_into_array(item)

        stack[:, :, itr] = current.image_array

    final_image = Image()
    final_image.image_array = np.std(stack, axis=2)
    final_image.write_array_into_image_file(filename, format)



def superposition_average():
    pass

"""
TODO: delete commented block
def speckle_contrast_amp(amplitude, roi = []):
    if not roi:
        roi = select_roi(amplitude, "Select a ROI to compute the speckle contrast")

    selection = amplitude[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]
    return np.std(selection)/np.average(selection)

"""