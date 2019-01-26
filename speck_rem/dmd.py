# Code to handle DMD related operations

from speck_rem.holography import *


def main():
    pass

class Mask(Image):
    """
    Class to handle the pattern masks to be projected onto the DMD
    """
    def __init__(self, width=1920, height=1080, pitch=7.6e-6):
        self.width = width
        self.height = height
        self.pitch = pitch
        super(Mask, self).__init__()

    def plane_phase(self, period, theta):
        """
        Compute a plane wave phase with a specified angle with the horizontal. This gets represented in the tilt of the
        fringes produced.
        :return: numpy array with the phase
        """
        u, v = np.meshgrid(np.linspace(0, self.width, self.width, endpoint=False) + 1/2,
                          np.linspace(0, self.height, self.height, endpoint=False) + 1/2)

        phase = (u * np.round(np.cos(theta), decimals=2) + v * np.round(np.sin(theta), decimals=2))

        return  (2.0 * math.pi / period) * phase

    def compute_plane_mask(self, period, theta):
        """
        Compute the mask for a tilted plane with an specific frequency and rotation
        :param period: in pixels
        :param theta: angle in radians
        :return: None
        """
        phase = self.plane_phase(period, theta)

        self.image_array = 1/2 + 1/2 * np.sign(np.sin(phase))

    def compute_fairness_constraint_mask(self, period, theta, pattern, grain):

        phase = self.plane_phase(period, theta)

        large_pattern = pattern.repeat(grain, axis=0).repeat(grain, axis=1)

        # WARNING: hard-coded to pad to DMD size
        large_pattern = np.pad(large_pattern, ((28, 28), (448, 448)), 'constant', constant_values=(0))

        self.image_array = 1 / 2 + 1 / 2 * np.sign(np.sin(phase - large_pattern*np.pi))

    def compute_random_mask(self, period, theta, grain):

        phase = self.plane_phase(period, theta)

        # WARNING: resizing is done with magic numbers
        window = int(np.ceil(1080/grain))

        pattern = np.random.randint(2, size=( window, window))

        large_pattern = pattern.repeat(grain, axis=0).repeat(grain, axis=1)

        large_pattern = large_pattern[0:1080, 0:1080]   # Forced-fix size inconsistencies
        large_pattern = np.pad(large_pattern, ((0, 0), (420,420)), 'constant', constant_values=(0))    # Pad to DMD size

        self.image_array = 1 / 2 + 1 / 2 * np.sign(np.sin(phase - large_pattern * np.pi))


if __name__ == "__main__":
    main()
