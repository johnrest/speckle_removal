#Compute the mask to project onto CCD

from speck_rem.holography import *

def main():
    pass


class Mask(Image):
    def __init__(self, width=1920, height=1080, pitch=7.6e-6):
        self.width = width
        self.height = height

        self.pitch = pitch
        super(Mask, self).__init__()

    def compute_mask(self, theta, period):
        """"
            Compute the mask for a tilted plane with an specific frequency and rotation
            Args:
                theta: rotation angle with respect to a horizontal
                period: Period in pixels
        """
        u, v = np.meshgrid(np.linspace(0, self.width, self.width, endpoint=False) + 1/2,
                          np.linspace(0, self.height, self.height, endpoint=False) + 1/2)

        temp = (u * np.round(np.cos(theta), decimals=2) + v * np.round(np.sin(theta), decimals=2))
        temp2 = np.sin((2.0 * math.pi / period) * temp)
        self.image_array = 1/2 + 1/2 * np.sign(temp2)


    def compute_fcn_mask(self,theta, period):
        grain = 32
        pattern_size = int(1024 / grain)
        number_pattern_images = (pattern_size * pattern_size) / 512
        pattern_batch = compute_pattern_batch(scale=pattern_size, batch_length=number_pattern_images)

        u, v = np.meshgrid(np.linspace(0, self.width, self.width, endpoint=False) + 1/2,
                          np.linspace(0, self.height, self.height, endpoint=False) + 1/2)

        temp = (u * np.round(np.cos(theta), decimals=2) + v * np.round(np.sin(theta), decimals=2))
        mod = pattern_batch[0]
        mod = mod.repeat(grain, axis=0).repeat(grain, axis=1)

        # print(np.pad(np.ones((3,3)), ((3, 2), (2, 3)), 'constant', constant_values=(0)))
        # pad1 = ((1080-1024)/2, (1080-1024)/2)
        mod = np.pad(mod, ((28, 28), (448,448)), 'constant', constant_values=(0))

        temp2 = np.sin((2.0 * math.pi / period) * temp - np.pi*mod)
        self.image_array = 1 / 2 + 1 / 2 * np.sign(temp2)


if __name__ == "__main__":
    main()
