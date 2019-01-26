from speck_rem.dmd import *

target_folder = "D:/Research/SpeckleRemoval/Data/2019_01_25/test"
mask_image_prefix = "pattern_"

# Select period values in pixels
# period_list = np.linspace(8, 14, num=4, endpoint=True)
period_list = [10.0]

# Select angle values
angle_list =  np.linspace(0.0, np.pi/2.0, num=3, endpoint=True)
# angle_list = [np.pi/4]

for itr, angle in enumerate(angle_list):
    for jtr, period in enumerate(period_list):
        mask = Mask()
        mask.compute_plane_mask(period, angle)
        print("Angle (grad): {0}; Period (pix): {1}".format(angle*180/np.pi, period))
        current_image_file = os.path.join(target_folder, mask_image_prefix + "{:03d}".format(itr*len(period_list)+jtr))
        print("Writing image to file: ", current_image_file, )
        mask.write_array_into_image_file(current_image_file, ".tiff")

# fig, ax = plt.subplots()
# im = ax.imshow(mask.image_array, origin='lower')
# fig.colorbar(im)
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()

