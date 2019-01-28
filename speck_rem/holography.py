# File to handle all holography processes
# All units in meters

# import speck_rem
from speck_rem import *
from speck_rem.admin import *

def main():
    pass

class Image:
    """Class to handle general image operations"""

    def __init__(self):
        self.image_array = np.array([])

    def read_image_file_into_array(self, filename):
        """
        Read image from file and create anp array
        :param filename: string with full filename
        :return: None
        """
        image_array = p_Image.open(filename)
        self.image_array = np.array(image_array)

        # Padding to produce squared images
        h, w = self.image_array.shape
        self.image_array = np.pad(self.image_array, ((int((w-h)/2), int((w-h)/2)), (0, 0)), 'reflect')

    def write_array_into_image_file(self, filename, format):
        """
        Write np array into an image of specified format, writes the abs if the input is complex
        :param filename: string with full filename
        :param format: string with the file format
        :return: None
        """
        if np.iscomplex(self.image_array).any():
            image = array_to_image(np.abs(self.image_array))
        else:
            image = array_to_image(self.image_array)

        cv2.imwrite(filename+format, image, [cv2.IMWRITE_PNG_BILEVEL, 1])

    def write_phase_into_image_file(self, filename, format):
        """
        Write the phase for a np array into an image of specified format
        :param filename:  string with full filename
        :param format: string with the file format
        :return: None
        """
        image = array_to_image(np.angle(self.image_array))
        cv2.imwrite(filename + format, image, [cv2.IMWRITE_PNG_BILEVEL, 1])

    def display(self):
        """
        Display the image through pillow
        :return: None
        """
        image = p_Image.fromarray(self.image_array, 'L')
        image.show()


class Hologram(Image):
    """Class to handle hologram operations"""

    def __init__(self, wavelength=473e-9, sensor_width=7.07e-3, sensor_height=5.3e-3):
        self.wavelength = wavelength
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        super(Hologram, self).__init__()

    def display_spectrum(self):
        """
        Compute and display the abs value of the spectrum of the hologram
        :return: None
        """
        fourier = np.fft.fft2(self.image_array, norm='ortho')
        spectrum = np.log(abs(fourier))
        display_image(spectrum, 0.5, "Spectrum")


class Reconstruction(Image):
    def __init__(self, hologram: Hologram, distance=0.0, spectrum_roi=None):
        self.distance = distance
        self.sensor_width = hologram.sensor_width
        self.sensor_height = hologram.sensor_height
        self.wavelength = hologram.wavelength

        self.pixel_width = self.distance*self.wavelength/self.sensor_width
        self.pixel_height = self.distance * self.wavelength / self.sensor_height
        self.spectrum_roi = spectrum_roi

        super(Reconstruction, self).__init__()

    def filter_hologram(self, hologram: Hologram):
        """
        Reconstruction with filter object wave with spectrum selection
        :param hologram: Hologram object
        :return: None
        """
        fourier = np.fft.fftshift(np.fft.fft2(hologram.image_array, norm='ortho'))
        spectrum = np.log(np.abs(fourier))
        spectrum = array_to_image(spectrum)

        if self.spectrum_roi is None:
            # Select ROI area and press enter for continuing
            self.spectrum_roi = select_roi(spectrum, "Select a ROI to filter the Hologram")

        r = self.spectrum_roi
        fourier_cropped = fourier[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        full_height, full_width = hologram.image_array.shape
        cropped_height, cropped_width = fourier_cropped.shape

        # Padding with zeros to expand to original size
        filtered = np.zeros((full_height, full_width), dtype=complex)
        init_width = int(full_width/2 - round(cropped_width/2))
        init_height = int(full_height/2 - round(cropped_height/2))
        filtered[init_height:init_height+cropped_height, init_width:init_width+cropped_width] = fourier_cropped

        # Inverse fourier transform to recover object wavefront
        object = np.fft.ifft2(filtered)
        self.image_array = object

    def propagate(self, distance):
        """
        Fresnel propagator
        :param distance: propagation distance in meters
        :return: None
        """
        self.distance = distance
        k = 2*math.pi/self.wavelength

        self.pixel_width = distance * self.wavelength / self.sensor_width
        self.pixel_height = distance * self.wavelength / self.sensor_height

        full_height, full_width = self.image_array.shape

        U, V = np.meshgrid(np.linspace(1, full_width, full_width),
                          np.linspace(1, full_height, full_height)-full_height/2)

        H = np.sqrt(np.power(U - full_width/2 - 1, 2) + np.power(V - full_height/2 - 1, 2))
        H = H * (self.sensor_width/full_width)
        H = np.exp(-1j*math.pi*(1/(self.wavelength*self.distance))*np.power(H,2))

        self.image_array = (np.fft.ifft2(np.fft.fftshift( self.image_array * H )))

class RandomPhaseMask(Image):
    """ Phase mask to combine with the hologram reconstruction"""

    def __init__(self, image_width=1280, image_height=1280):
        self.image_width = image_width
        self.image_height = image_height

        super(RandomPhaseMask, self).__init__()

    def create(self, grain=1280/16):
        """
        Create the mask with a specified block size
        :param grain: in pixels
        :return: None
        """
        full_width, full_height = (self.image_width, self.image_height)
        scale = int(full_width/grain)
        modulation_array = np.exp(1j * np.pi * np.random.randint(2, size=(scale,scale)))        #phase
        # modulation_array = np.random.randint(2, size=(scale,scale))                           #amplitude
        self.image_array = modulation_array.repeat(grain, axis=0).repeat(grain, axis=1)

    def optimize(self, grain=1280/4, fraction_set=0.01):
        """
        Compute a random binary phase mask, with a reduced number of possible black pixels. This reduce the undesired
        effects on the reconstructed image
        :param grain: in pixels
        :param fraction_set: value in range (0,1]
        :return: None
        """
        shape_set = (np.round(np.sqrt(self.image_width*self.image_height*fraction_set)), )*2
        shape_set = tuple(map(lambda x: int(x), shape_set))
        off_positions = np.random.randint(self.image_width, size=(2, shape_set[0]*shape_set[1]))

        modulation_array = np.ones((self.image_height, self.image_width))
        modulation_array[off_positions[0,:], off_positions[1,:]] = 0.0
        modulation_array = np.exp(1j * np.pi * modulation_array)            # phase
        self.image_array = modulation_array


class FairnessConstraintMask(Image):
    """Phase mask under fairness constratint sampling to use in hologram reconstruction"""

    def __init__(self, image_width=1280, image_height=1280):
        self.image_width = image_width
        self.image_height = image_height

        super(FairnessConstraintMask, self).__init__()

    def compute(self, grain=1280/4, pattern=np.ones((1,1))):
        """
        Mask is computed from smaller pattern obtained by a separate funtion
        :param grain: in pixels
        :param pattern: numpy array with the sampling
        :return: None
        """
        modulation_array = np.exp(1j * np.pi * pattern)        #phase
        # modulation_array = np.random.randint(2, size=(scale,scale))                           #amplitude
        self.image_array = modulation_array.repeat(grain, axis=0).repeat(grain, axis=1)


if __name__ == "__main__":
    main()
