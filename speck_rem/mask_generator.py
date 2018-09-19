#Compute the mask to project onto CCD

from speck_rem.main import *


def main():
    mask = Mask()
    mask.compute_mask(math.pi/2, 16)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


class Mask(Image):
    def __init__(self, width=1920, height=1080, pitch=7.6e-6):
        self.width = width
        self.height = height

        self.pitch = pitch
        super(Mask, self).__init__()

    def compute_mask(self, theta, frequency):
        """"
            Compute the mask for a tilted plane with an specific frequency and rotation
            Args:
                theta: rotation angle with respect to a horizontal
                frequency: Spatial frequency in pixels
        """
        u, v = np.meshgrid(np.linspace(1, self.width, self.width) - self.width/2,
                          np.linspace(1, self.height, self.height)-self.height)

        dmd_pattern = 1/2 + 1/2 * np.sign(np.cos((2*math.pi/frequency)*(u*math.cos(theta) + math.sin(theta)*v)))

        fig, ax = plt.subplots()
        im = ax.imshow(dmd_pattern, origin='lower')
        fig.colorbar(im)
        plt.show()


if __name__ == "__main__":
    main()
