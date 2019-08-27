# Tools related to speckle computations

from speck_rem import *


def speckle_correlation_coefficient(images_list, roi=True):
    """
    Compute the speckle correlation coefficient for a list of reconstructed images
    :param images_list: list of filenames as strings
    :param roi: flag to indicate if roi needs to be selected. True for roi selection, false for full image
    :return: numpy 2D array with the coefficients
    """
    first_image = Image()
    first_image.read_image_file_into_array(images_list[0])

    if roi is True:
        roi = select_roi(first_image.image_array, "Select a ROI to compute the speckle correlation")
    else:
        roi = [0, 0, *first_image.image_array.shape]

    speckle_corr_coeff = np.empty((len(images_list), len(images_list)), dtype=float)

    for ii, file_ii in enumerate(images_list):
        image_ii = Image()
        image_ii.read_image_file_into_array(file_ii)
        amp_ii = image_ii.image_array[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        for jj, file_jj in enumerate(images_list):
            image_jj = Image()
            image_jj.read_image_file_into_array(file_jj)
            amp_jj = image_jj.image_array[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            speckle_corr_coeff[ii, jj] = np.abs(np.sum((amp_ii - np.mean(amp_ii)) * (amp_jj - np.mean(amp_jj)))) \
                                         / np.sqrt(
                np.sum(np.power(amp_ii - np.mean(amp_ii), 2)) * np.sum(np.power(amp_jj - np.mean(amp_jj), 2)))
        print("...")

    return speckle_corr_coeff


def speckle_metrics(images_list, roi=None):
    """ Compute different metrics for the speckle noise, under two modalities
        of superposition: average and standard deviation.
        - Speckle contrast SC
        - Speckle suppression index SSI
        - Speckle suppression and mean preservation index   SMPI
        Creates an image stack with a ROI selection to avoid memory issues.
        :param list with coordinates for roi or empty if roi needs to be selected
        :return tuple with all speckle coefficients paired up for average and std modes
        """

    first_image = Image()
    first_image.read_image_file_into_array(images_list[0])

    if not roi:
        roi = select_roi(first_image.image_array, "Select a ROI to compute the speckle contrast")

    stack = np.empty((roi[3], roi[2], len(images_list)), dtype=float)

    sc_avg, sc_std, ssi_avg, ssi_std, smpi_avg, smpi_std = [], [], [], [], [], []

    for itr, item in enumerate(images_list):
        current = Image()
        current.read_image_file_into_array(item)

        stack[:, :, itr] = current.image_array[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]

        average_image = np.average(stack[:, :, 0:(itr + 1)], axis=2)

        standard_dev_image = np.std(stack[:, :, 0:(itr + 1)], axis=2)

        # print(np.shape(stack[:, :, 0:(itr + 1)]))

        sc_avg.append(np.std(average_image) / np.average(average_image))
        sc_std.append(np.std(standard_dev_image) / np.average(standard_dev_image))

        ssi_avg.append((np.std(average_image) / np.average(average_image)) *
                       (np.average(stack[:, :, 0]) / np.std(stack[:, :, 0])))
        ssi_std.append((np.std(standard_dev_image) / np.average(standard_dev_image)) *
                       (np.average(stack[:, :, 0]) / np.std(stack[:, :, 0])))

        smpi_avg.append((1 + np.abs(np.average(average_image) - np.average(stack[:, :, 0]))) *
                        (np.std(average_image) / np.std(stack[:, :, 0])))

        smpi_std.append((1 + np.abs(np.average(standard_dev_image) - np.average(stack[:, :, 0]))) *
                        (np.std(standard_dev_image) / np.std(stack[:, :, 0])))

        # harmonic_average = (itr + 1)/np.sum(1/stack)

    return sc_avg, sc_std, ssi_avg, ssi_std, smpi_avg, smpi_std


def superposition_standard_dev(images_list, filename, format):
    """
    Compute the superposition image as the axial standard deviation of the stack pixel by pixel, write to a file
    :param images_list: list of full filenames
    :param filename: full filename to write the final superposed image
    :return None
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


def superposition_average(images_list, filename, format):
    """
    Compute the superposition image as the average of the reconstructed images stack
    :param images_list: List of image file names
    :param filename: full output file name to store the result
    :param format: file type for the result
    :return: None
    """
    first_image = Image()
    first_image.read_image_file_into_array(images_list[0])

    array = np.ones(first_image.image_array.shape, dtype=float)

    for itr, item in enumerate(images_list):
        current = Image()
        current.read_image_file_into_array(item)
        array += current.image_array

    final_image = Image()
    final_image.image_array = array / len(images_list)  # compute average
    final_image.write_array_into_image_file(filename, format)
