#Compute the mask to project onto CCD
from prog import *
from prog.main import *

def main():
    Phi = Mask()
    Phi.compute_mask(math.pi/2.0, math.pi/2.0, 1/16)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



class Mask(Image):
    def __init__(self, width=1920, height=1080, pitch=7.6e-6):
        self.width = width
        self.height = height
        self.pitch = pitch
        super(Mask, self).__init__()


    def compute_mask(self, alpha, beta, frequency):
        U,V = np.meshgrid(np.linspace(1, self.width, self.width) - self.width/2,
                          np.linspace(1, self.height, self.height)-self.height)

        phase_pattern = (U*math.cos(alpha) + V*math.cos(beta))\
                        /(1 - np.power(math.cos(alpha),2) - np.power(math.cos(beta),2))

        dmd_pattern = (1 + np.cos(2*math.pi*(U-V)*frequency/2 - phase_pattern))*1/2

        dmd_pattern = np.where(dmd_pattern>0.5, 1.0, 0.0)


        fig, ax = plt.subplots()
        im = ax.imshow(dmd_pattern, origin='lower')
        fig.colorbar(im)
        plt.show()



if __name__ == "__main__":
    main()
