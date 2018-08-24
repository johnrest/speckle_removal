# main file
# All units in meters
import numpy as np
from matplotlib import pyplot as plt
import os, glob
from PIL import Image as p_Image
import math

def main():
    print("SPECKLE REMOVAL PROJECT")
    target_folder = "C:/Users/itm/Desktop/DH/2018_08_17/dice_16_walsh"
    target_mask = "holo*"

    images_list = get_list_images(target_folder, target_mask)

    holo = Hologram()
    holo.read_image_file_into_array(images_list[0])
    holo.display_spectrum()



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
        spectrum *= 255.0 / spectrum.max()
        #spectrum = np.uint8(spectrum)
        print(spectrum.max())
        image = p_Image.fromarray( spectrum, 'L')
        image.show()
        #TODO: Spectrum is not displaying the correct image. fix: Pillow show is using a windows application

if __name__ == "__main__":
    main()
