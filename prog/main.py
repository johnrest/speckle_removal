# main file
# All units in meters
import numpy as np
from matplotlib import pyplot as plt
import os, glob
from PIL import Image as p_Image
import cv2
import math

def main():
    print("SPECKLE REMOVAL PROJECT")
    target_folder = "C:/Users/itm/Desktop/DH/2018_08_17/dice_16_walsh"
    target_mask = "holo*"

    images_list = get_list_images(target_folder, target_mask)

    holo = Hologram()
    holo.read_image_file_into_array(images_list[0])
    # holo.display_spectrum()

    recon = Reconstruction(holo)

    # plt.show()
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
        """Write np array into an image of specified format"""
        #TODO:implement the writing into file
        pass

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
        spectrum = np.fft.fft2(self.image_array, norm='ortho')
        spectrum =  np.log(abs(spectrum))
        spectrum -= spectrum.min()
        spectrum *= 255 / spectrum.max()
        imS = spectrum.astype(np.uint8)
        imS = cv2.resize(imS, (640, 512))
        cv2.imshow("Spectrum", imS)
        r = cv2.selectROI(spectrum.astype(np.uint8))
        imCrop = spectrum[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cv2.imshow("Image", np.uint8(imCrop))
        # TODO: continue from this crop to the reconstruction of the hologram.


class Reconstruction(Image):
    def __init__(self, holo: Hologram, distance=0, spectrum_roi=None):
        self.distance = distance
        super(Reconstruction, self).__init__()

        if spectrum_roi == None:
            self.filter_hologram(holo)
        else:
            self.spectrum_roi = spectrum_roi

    def filter_hologram(self, holo: Hologram):
        holo.display_spectrum()


if __name__ == "__main__":
    main()
