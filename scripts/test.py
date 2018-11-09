# Testing script for testing the tests
from speck_rem import *


target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/test"
mask_image_prefix = "pattern_"

mask = Mask()
mask.compute_random_mask(np.pi/4.0, 16, 16)

display_image(mask.image_array, 0.5, "mask")


cv2.waitKey(0)
cv2.destroyAllWindows()

