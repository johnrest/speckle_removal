from speck_rem import *
from speck_rem.dmd import *

target_folder = "C:/Users/itm/Desktop/DH/2018_10_31/test_random_patternC"
mask_image_prefix = "pattern_"
number_patterns = 15
grain = 16

# Select period values in pixels
# period_list = np.linspace(8, 20, num=7, endpoint=True)
period = 10.0

# Select angle values
# angle_list =  np.linspace(0.0, np.pi/2.0, num=5, endpoint=True)
angle = np.pi/4

for itr, item in enumerate(range(number_patterns)):
    mask = Mask()
    mask.compute_random_mask(angle, period, grain)
    print("Angle (grad): {0}; Period (pix): {1}".format(angle * 180 / np.pi, period))
    current_image_file = os.path.join(target_folder, mask_image_prefix + "{:03d}".format(itr))
    print("Writing image to file: ", current_image_file, )
    mask.write_array_into_image_file(current_image_file, ".png")