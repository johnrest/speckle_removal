#Compute the mask to project onto CCD
from main import Image, display_image

def main():
    pass

class Mask(Image):
    def __init__(self, width=1920, height=1080, pitch=7.6e-6):
        self.width = width
        self.height = height
        self.pitch = pitch
        super(Mask, self).__init__()


    def compute_mask(self, alpha, beta):
        pass
        # U,V = np.meshgrid(np.linspace(1, full_width, full_width),
        #                   np.linspace(1, full_height, full_height)-full_height/2)
        #
        # H = np.sqrt(np.power(U - full_width/2 - 1, 2) + np.power(V - full_height/2 - 1, 2))

if __name__ == "__main__":
    main()
