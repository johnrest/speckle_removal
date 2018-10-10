#Compute the mask to project onto CCD

from speck_rem.holography import *

def main():
    target_folder = "C:/Users/itm/Desktop/DH/2018_10_05/test"

    mask_image_prefix = "pattern_"

    #angle_list =  np.linspace(0.0, np.pi/2.0, num=5, endpoint=True)
    angle = math.pi/2.0

    #period = 20
    period_list =  np.linspace(8, 20, num=7, endpoint=True)

    for itr, a in enumerate(period_list):
        mask = Mask()
        mask.compute_mask(angle, a)
        print(a)
        current_image_file = os.path.join(target_folder, mask_image_prefix + "{:03d}".format(itr))
        print( "Writing image to file: ", current_image_file, )
        mask.write_array_into_image_file(current_image_file, ".png")

    # fig, ax = plt.subplots()
    # im = ax.imshow(mask.image_array, origin='lower')
    # fig.colorbar(im)
    # plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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
        # u, v = np.meshgrid(np.linspace(0, self.width, self.width, endpoint=False) - (self.width/2-1) - 1/2,
        #                   np.linspace(0, self.height, self.height, endpoint=False) - (self.height/2-1) - 1/2)

        u, v = np.meshgrid(np.linspace(0, self.width, self.width, endpoint=False) + 1/2,
                          np.linspace(0, self.height, self.height, endpoint=False) + 1/2)


        #_ = np.round(np.cos((2*np.pi/period) * (u*math.cos(theta) + v*math.sin(theta))), decimals=10)
        #self.image_array = (1/2) + (1/2) * np.where( _ >= 0, 1.0, -1.0)

        temp = (u * np.round(np.cos(theta), decimals=2) + v * np.round(np.sin(theta), decimals=2))
        temp2 = np.cos((2.0 * math.pi / period) * temp)
        self.image_array = 1/2 + 1/2 * np.sign(temp2)

        #self.image_array = 1/2 + 1/2 * np.sign(np.cos((2 * np.pi / period) * (u * np.cos(theta) + v * np.sin(theta))))


if __name__ == "__main__":
    main()
