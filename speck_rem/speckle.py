# Tools related to speckle computations

from speck_rem import *


def speckle_correlation_coefficient(image_batch, roi=True):

    if roi is True:
        roi = select_roi(np.abs(image_batch[0].image_array), "Select a ROI to compute the speckle correlation")
    else:
        roi = image_batch[0].image_array.shape              #Compute over full image


    cc_speckle = np.empty((len(image_batch), len(image_batch)),dtype=float)

    for ii, image_p in enumerate(image_batch):
        for jj, image_q in enumerate(image_batch):
            Ip = np.abs(image_p.image_array[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])])
            Iq = np.abs(image_q.image_array[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])])
            cc_speckle[ii, jj] = np.abs(np.sum((Ip-np.mean(Ip))*(Iq-np.mean(Iq))))\
                                 /np.sqrt(np.sum(np.power(Ip-np.mean(Ip), 2)) * np.sum(np.power(Iq-np.mean(Iq), 2)))
            print(cc_speckle[ii, jj])

    return cc_speckle

def speckle_contrast(imagebatch, roi=[]):

    result = []
    for itr, item in enumerate(imagebatch):
        if not roi:
            roi = select_roi(np.abs(item.image_array), "Select a ROI to compute the speckle contrast")

        selection = np.abs(item.image_array)[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]

        result.append(np.std(selection)/np.average(selection))

    return result

def speckle_contrast_amp(amplitude, roi = []):
    if not roi:
        roi = select_roi(amplitude, "Select a ROI to compute the speckle contrast")

    selection = amplitude[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]
    return np.std(selection)/np.average(selection)
