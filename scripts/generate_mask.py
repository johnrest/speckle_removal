from speck_rem.dmd import *

target_folder = "C:/Users/itm/Desktop/DH/2018_10_25/test"
mask_image_prefix = "pattern_"

# Select period values in pixels
# period_list = np.linspace(8, 20, num=7, endpoint=True)
period_list = [10.0]

# Select angle values
# angle_list =  np.linspace(0.0, np.pi/2.0, num=5, endpoint=True)
angle_list = [np.pi/4]

for itr, angle in enumerate(angle_list):
    for jtr, period in enumerate(period_list):
        mask = Mask()
        # mask.compute_mask(angle, period)
        mask.compute_fcn_mask(angle, period)
        print("Angle (grad): {0}; Period (pix): {1}".format(angle*180/np.pi, period))
        current_image_file = os.path.join(target_folder, mask_image_prefix + "{:03d}".format(itr+jtr))
        print("Writing image to file: ", current_image_file, )
        mask.write_array_into_image_file(current_image_file, ".png")

# fig, ax = plt.subplots()
# im = ax.imshow(mask.image_array, origin='lower')
# fig.colorbar(im)
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()

