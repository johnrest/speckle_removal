# main file
# All units in meters
# import numpy as np
# from matplotlib import pyplot as plt
# import os, glob
# from PIL import Image as p_Image
# import cv2
# import math

from speck_rem import *

def main():
    print("SPECKLE REMOVAL PROJECT")
    target_folder = "C:/Users/itm/Desktop/DH/2018_09_21/dice_rot_fixed_freq"
    target_filename = "holo"
    reconstruct_prefix = "rec_"
    focusing_distance = 1.3         #1.7 for dice rotating / 1.3 for dice walsh
    recon_batch = list()

    extract_frames_from_video(target_folder, "holo.avi", target_filename)

    images_list = get_list_images(target_folder, target_filename+"_0*")

    for itr, item in enumerate(images_list):

        holo = Hologram()
        holo.read_image_file_into_array(item)

        if itr == 0:
            recon = Reconstruction(holo)
        else:
            recon = Reconstruction(holo, spectrum_roi=selected_roi)


        recon.filter_hologram(holo)
        selected_roi = recon.spectrum_roi
        prop = recon
        prop.image_array = recon.propagate(focusing_distance)
        recon_batch.append(prop)

        prop.write_array_into_image_file(os.path.join(target_folder, reconstruct_prefix+"{:02d}".format(itr)), ".bmp")
        print("Copying to image: " + os.path.join(target_folder, reconstruct_prefix+"{:02d}".format(itr)), ".bmp")

    # display_image(abs(prop.image_array), 0.5, "Propagated amplitude")

    speckle_correlation_coefficient(recon_batch, roi=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_list_images(directory, mask):
    "List of files containing the mask pattern within directory"
    return glob.glob(os.path.join(directory, mask))


class Image:
    """Main class to handle general image operations"""

    def __init__(self):
        self.image_array = np.array([])

    def read_image_file_into_array(self, filename):
        """Read image from file and create anp array"""
        image_array = p_Image.open(filename)
        self.image_array = np.array(image_array)

    def write_array_into_image_file(self, filename, format):
        """Write np array into an image of specified format,
        writes the abs if the input is complex"""
        if np.iscomplex(self.image_array).any():
            image = array_to_image(np.abs(self.image_array))
        else:
            image = array_to_image(self.image_array)

        cv2.imwrite(filename+format, image, [cv2.IMWRITE_PNG_BILEVEL, 1])


    def display(self):
        """Display the image through pillow"""
        image = p_Image.fromarray(self.image_array, 'L')
        image.show()

class Hologram(Image):
    """Class to handle holograms"""

    def __init__(self, wavelength=473e-9, sensor_width=6.78e-3, sensor_height=5.43e-3):
        self.wavelength = wavelength
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        super(Hologram, self).__init__()

    def display_spectrum(self):
        fourier = np.fft.fft2(self.image_array, norm='ortho')
        spectrum =  np.log(abs(fourier))
        display_image(spectrum, 0.5, "Spectrum")

class Reconstruction(Image):
    def __init__(self, holo: Hologram, distance=0.0, spectrum_roi=None):
        self.distance = distance
        self.sensor_width = holo.sensor_width
        self.sensor_height = holo.sensor_height
        self.wavelength = holo.wavelength

        self.pixel_width = self.distance*self.wavelength/self.sensor_width
        self.pixel_height = self.distance * self.wavelength / self.sensor_height
        self.spectrum_roi = spectrum_roi

        super(Reconstruction, self).__init__()

    def filter_hologram(self, holo: Hologram):
        fourier = np.fft.fftshift(np.fft.fft2(holo.image_array, norm='ortho'))
        spectrum = np.log(abs(fourier))
        spectrum = array_to_image(spectrum)

        windowName = "Select filter and press Enter"
        if self.spectrum_roi is None:
            # Select area and press enter for continuing
            self.spectrum_roi = cv2.selectROI(img=spectrum, windowName=windowName)
        cv2.destroyWindow(windowName)

        r = self.spectrum_roi
        fourier_cropped = fourier[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        full_height, full_width = holo.image_array.shape
        cropped_height, cropped_width = fourier_cropped.shape

        # Padding with zeros to expand to original size
        filtered = np.zeros((full_height, full_width), dtype=complex)
        init_width = int(full_width/2 - round(cropped_width/2))
        init_height = int(full_height/2 - round(cropped_height/2))
        filtered[init_height:init_height+cropped_height, init_width:init_width+cropped_width] = fourier_cropped

        #Inverse fourier transform to recover object wavefront
        object = np.fft.ifft2(filtered)
        self.image_array = object


    def propagate(self, distance):
        self.distance = distance
        k = 2*math.pi/self.wavelength

        #Image pixel sizes
        self.pixel_width = distance * self.wavelength / self.sensor_width
        self.pixel_height = distance * self.wavelength / self.sensor_height

        full_height, full_width = self.image_array.shape

        U,V = np.meshgrid(np.linspace(1, full_width, full_width),
                          np.linspace(1, full_height, full_height)-full_height/2)

        H = np.sqrt(np.power(U - full_width/2 - 1, 2) + np.power(V - full_height/2 - 1, 2))
        H = H*(self.sensor_width/full_width)
        H = np.exp(-1j*math.pi*(1/(self.wavelength*self.distance))*np.power(H,2))

        propagated = (np.fft.ifft2(np.fft.fftshift( self.image_array * H )))
        return propagated

# Helper functions
def display_image(array, scale=1, title="Image"):
    image = array_to_image(array)
    height, width = image.shape
    height = int(height*scale)
    width = int(width * scale)
    resampled = cv2.resize(image, (width, height))
    cv2.imshow(title, resampled)


def array_to_image(array):
    array = array.astype(np.float64)
    array -= array.min()
    array *= 255 / array.max()
    image = array.astype(np.uint8)
    return image


def speckle_correlation_coefficient(image_batch, roi=True):

    # if roi is True:
        # Select area and press enter for continuing
        # windowName = "Select ROI and press Enter"
        # display_image(np.abs(image_batch[2].image_array), 0.5, "recon")
        # r = cv2.selectROI(img=np.abs(image_batch[2].image_array), windowName=windowName, fromCenter=False)

    cc_speckle = np.empty((len(image_batch), len(image_batch)),dtype=float)
    for ii, image_p in enumerate(image_batch):
        for jj, image_q in enumerate(image_batch):
            Ip = np.abs(image_p.image_array)
            Iq = np.abs(image_q.image_array)
            # Ip = np.abs(image_p.image_array[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
            # Iq = np.abs(image_q.image_array[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
            cc_speckle[ii, jj] = np.abs(np.sum((Ip-np.mean(Ip))*(Iq-np.mean(Iq))))\
                                 /np.sqrt(np.sum(np.power(Ip-np.mean(Ip), 2)) * np.sum(np.power(Iq-np.mean(Iq), 2)))
            print(cc_speckle[ii, jj])

    fig, ax = plt.subplots()
    im = ax.imshow(cc_speckle, origin='lower')
    fig.colorbar(im)
    plt.show()

    # TODO: attempt for ROI selection now working
    # TODO: helper function to compute speckle contrast


def extract_frames_from_video(target_folder, video_filename, image_name_mask):
    """ Extract all frames from a video and store as png files in the same folder"""

    video_capture = cv2.VideoCapture(os.path.join(target_folder, video_filename))
    success, image = video_capture.read()
    print("First frame read", success)
    count = 0
    while success:
        fname = os.path.join(target_folder, image_name_mask+"_{:03d}".format(count)+".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite( fname, image)  # save frame as PNG file
        success, image = video_capture.read()
        print("Reading next frame: ", success)
        count += 1

if __name__ == "__main__":
    main()

