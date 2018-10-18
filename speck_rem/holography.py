# File to handle all holography processes
# All units in meters

# import speck_rem
from speck_rem import *
from speck_rem.admin import *

def main():
    pass

class Image:
    """Main class to handle general image operations"""

    def __init__(self):
        self.image_array = np.array([])

    def read_image_file_into_array(self, filename):
        """Read image from file and create anp array"""
        image_array = p_Image.open(filename)
        self.image_array = np.array(image_array)

        # Padding to produce holograms
        h, w = self.image_array.shape
        self.image_array = np.pad(self.image_array, ((int((w-h)/2), int((w-h)/2)), (0, 0)), 'reflect')


    def write_array_into_image_file(self, filename, format):
        """Write np array into an image of specified format,
        writes the abs if the input is complex"""
        if np.iscomplex(self.image_array).any():
            image = array_to_image(np.abs(self.image_array))
            # image = array_to_image(np.angle(self.image_array))
        else:
            image = array_to_image(self.image_array)

        cv2.imwrite(filename+format, image, [cv2.IMWRITE_PNG_BILEVEL, 1])

    def write_phase_into_image_file(self, filename, format):
        """Write the phase for a np array into an image of specified format"""
        image = array_to_image(np.angle(self.image_array))
        cv2.imwrite(filename + format, image, [cv2.IMWRITE_PNG_BILEVEL, 1])

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
        spectrum = np.log(np.abs(fourier))
        spectrum = array_to_image(spectrum)


        if self.spectrum_roi is None:
            # Select area and press enter for continuing
            self.spectrum_roi = select_roi(spectrum, "Select a ROI to compute the speckle contrast")

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


class RandomPhaseMask(Image):
    def __init__(self, image_width=1280, image_height=1280):
        self.image_width = image_width
        self.image_height = image_height

        super(RandomPhaseMask, self).__init__()

    def create(self, grain=1280/16):
        # modulation_array = np.exp(-1j*np.pi*np.random.randint(2, size=self.image_array.shape))
        full_width, full_height = (self.image_width, self.image_height)
        scale = int(full_width/grain)
        modulation_array = np.exp(1j * np.pi * np.random.randint(2, size=(scale,scale)))        #phase
        # modulation_array = np.random.randint(2, size=(scale,scale))                           #amplitude
        self.image_array = modulation_array.repeat(grain, axis=0).repeat(grain, axis=1)

    def optimize(self, grain=1280/4, fraction_set=0.01):
        """Compute a random binary phase mask, with a reduced number of possible black pixels"""
        print(np.round(np.sqrt(self.image_width*self.image_height*fraction_set)))
        shape_set = (np.round(np.sqrt(self.image_width*self.image_height*fraction_set)), )*2
        shape_set = tuple(map(lambda x: int(x), shape_set))
        print(shape_set)
        off_positions = np.random.randint(self.image_width+1, size=shape_set)
        modulation_array = np.ones((self.image_height, self.image_width))
        modulation_array[off_positions] = 0.0


if __name__ == "__main__":
    main()
