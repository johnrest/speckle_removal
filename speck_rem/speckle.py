# Tools related to speckle computations

import speck_rem
from speck_rem import *

def speckle_correlation_coefficient(image_batch, roi=True):

    if roi is True:
        # Select area and press enter for continuing
        # windowName = "Select ROI and press Enter"
        # display_image(np.abs(image_batch[2].image_array), 0.5, "recon")
        # r = cv2.selectROI(img=np.abs(image_batch[2].image_array), windowName=windowName, fromCenter=False)
        r = [400,300,450,320]
        # Ip = np.abs(np.abs(image_batch[0].image_array)[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
        # display_image(Ip, 0.5, "image")
        #TODO: fix issue with ROI selection

    cc_speckle = np.empty((len(image_batch), len(image_batch)),dtype=float)

    for ii, image_p in enumerate(image_batch):
        for jj, image_q in enumerate(image_batch):
            # Ip = np.abs(image_p.image_array)
            # Iq = np.abs(image_q.image_array)
            Ip = np.abs(image_p.image_array[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
            Iq = np.abs(image_q.image_array[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
            cc_speckle[ii, jj] = np.abs(np.sum((Ip-np.mean(Ip))*(Iq-np.mean(Iq))))\
                                 /np.sqrt(np.sum(np.power(Ip-np.mean(Ip), 2)) * np.sum(np.power(Iq-np.mean(Iq), 2)))
            print(cc_speckle[ii, jj])

    fig, ax = plt.subplots()
    im = ax.imshow(cc_speckle, origin='lower')
    fig.colorbar(im)
    plt.show()

    # TODO: attempt for ROI selection now working
    # TODO: helper function to compute speckle contrast