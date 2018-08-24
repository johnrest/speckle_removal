# main file
# All units in meters
import numpy as np
from matplotlib import pyplot as plt
import os, glob
from PIL import Image as p_Image


def main():
    pass


def get_list_images(directory, mask):
    "List of files containing the mask pattern within directory"
    return glob.glob(os.path.join(directory, mask))


class Image:
    """Main class to handle general image operations"""

    def __init__(self):
        self.image_array = np.array()

    def read_image_file_into_array(self, filename):
        """Read image from file and create anp array"""
        image_array = p_Image.open(filename)
        return np.array(image_array)

    def write_array_into_image_file(self, filename, format):
        """Write np array into an image of specified format"""
        pass


class Hologram(Image):
    """Class to handle holograms"""

    def __init__(self, wavelength=473e-9, sensor_width=6.78e-3, sensor_height=5.43e-3):
        self.wavelength = wavelength
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        super(Hologram, self).__init__(Image)

    def display_spectrum(self):
        pass


if __name__ == "__main__":
    main()
