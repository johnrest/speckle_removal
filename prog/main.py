# main file
# All units in meters
import numpy as np
from matplotlib import pyplot as plt
import os, glob
from PIL import Image as p_Image
import cv2
import math
import cmath

def main():
    print("SPECKLE REMOVAL PROJECT")
    target_folder = "C:/Users/itm/Desktop/DH/2018_08_17/dice_16_walsh"
    target_mask = "holo*"

    images_list = get_list_images(target_folder, target_mask)

    holo = Hologram()
    holo.read_image_file_into_array(images_list[0])
    # holo.display_spectrum()

    recon = Reconstruction(holo)
    recon.propagate(1.3)

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
        fourier = np.fft.fft2(self.image_array, norm='ortho')
        spectrum =  np.log(abs(fourier))
        display_image(spectrum, 0.5, "Spectrum")

class Reconstruction(Image):
    def __init__(self, holo: Hologram, distance=0, spectrum_roi=None):
        self.distance = distance
        self.sensor_width = holo.sensor_width
        self.sensor_height = holo.sensor_height
        self.wavelength = holo.wavelength

        self.pixel_width = self.distance*self.wavelength/self.sensor_width
        self.pixel_height = self.distance * self.wavelength / self.sensor_height

        super(Reconstruction, self).__init__()

        if spectrum_roi == None:
            self.filter_hologram(holo)
        else:
            self.spectrum_roi = spectrum_roi

    def filter_hologram(self, holo: Hologram):
        fourier = np.fft.fftshift(np.fft.fft2(holo.image_array, norm='ortho'))
        spectrum = np.log(abs(fourier))
        spectrum = array_to_image(spectrum)

        self.spectrum_roi = cv2.selectROI(spectrum)
        r = self.spectrum_roi

        #Select area and press enter for continuing
        fourier_cropped = fourier[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        full_height, full_width  = holo.image_array.shape
        cropped_height, cropped_width = fourier_cropped.shape

        #Padding with zeros to expand to original size
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

        u = 1/((np.linspace(1, full_width, full_width)-full_width/2)*(self.sensor_width/full_width))
        v = 1/((np.linspace(1, full_height, full_height)-full_height/2)*(self.sensor_width/full_width))

        mask=np.zeros((full_height, full_width), dtype=complex)
        mask[500:524,620:660] = 1.0
        display_image(mask, 0.5, "Mask")

        U, V = np.meshgrid(u, v)

        O = np.fft.fftshift(np.fft.fft2(mask, norm='ortho'))
        display_image(np.abs(O), 0.5, "Abs of O")

        # H = np.exp(1j*k*self.distance) * np.exp(-1j*math.pi*self.wavelength*self.distance*(np.power(U,2) + np.power(V,2)))
        H = np.exp(1j * k * self.distance* np.sqrt( 1 - (np.power(U*self.wavelength, 2) + np.power(V*self.wavelength , 2))))

        U = np.fft.ifftshift(np.fft.ifft2(H, norm='ortho'))

        display_image(np.abs(U), 0.5, "Abs of U")

        #TODO: Fresnel transform not working




#Helper functions
def display_image(array, scale=1, title="Image"):
    image = array_to_image(array)
    height, width = image.shape
    height = int(height*scale)
    width = int(width * scale)
    resampled = cv2.resize(image, (width, height))
    cv2.imshow(title, resampled)


def array_to_image(array):
    array -= array.min()
    array *= 255 / array.max()
    image = array.astype(np.uint8)
    return image


if __name__ == "__main__":
    main()

