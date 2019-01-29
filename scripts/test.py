# Testing script for testing the tests
from speck_rem import *
from matplotlib import cm as CM                 #Move to ini file


target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/random_different_sized_grain/"
hologram_name_mask = "holo_0*"
images_list = get_list_images(target_folder, hologram_name_mask)

for itr, item in enumerate(images_list[1:]):
    print("Processing hologram :", item)
    print("... ... ...")

    hologram = Hologram()
    hologram.read_image_file_into_array(item)

    holo_sub = Hologram()
    holo_sub.read_image_file_into_array(images_list[itr-1])

    hologram.image_array = hologram.image_array - holo_sub.image_array
    hologram.image_array -= np.min(hologram.image_array)

    current_file = os.path.join(target_folder, "holo_" + "{:03d}".format(itr+20))

    hologram.write_array_into_image_file(current_file, ".tiff")


# # target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/three/planar/"
# target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/planar_fixed_freq_manual/"
# results_folder = create_folder(target_folder, "comp")
# holo_name_mask = "holo_*"
# reconstruct_prefix = "full_recon"
# results_folder = target_folder
# reconstruct_format = ".tiff"
#
#
# focusing_distance = 0.85                    # meters
# # focusing_distance = 1.5                    # meters
#
# images_list = get_list_images(target_folder, holo_name_mask)
#
# holo_total = np.zeros((2048, 2048), dtype=np.float32)
# # holo_total.read_image_file_into_array(images_list[0])
#
# for itr, item in enumerate(images_list):
#     holo = Hologram()
#     holo.read_image_file_into_array(item)
#     holo_total += holo.image_array
#
# holo.image_array = holo_total
# rec = Reconstruction(holo)
# rec.filter_hologram(holo)
# roi = rec.spectrum_roi
#
# rec.propagate(focusing_distance)
#
#
# display_image(holo.image_array, 0.5, "holo")
# display_image(np.abs(rec.image_array), 0.5, "Rec")
#
# current_file = os.path.join(results_folder, reconstruct_prefix)
# print("Copying image to file: " + current_file + reconstruct_format)
# print("... ... ...")
# rec.write_array_into_image_file(current_file, reconstruct_format)
# crop_image(current_file + reconstruct_format, current_file + reconstruct_format)
#
# cv2.waitKey()

# target_folder = "C:/Users/itm/Desktop/DH/2018_11_22/three/random_different_sized_grain/"
# results_folder = create_folder(target_folder, "comp")
# holo_name_mask = "holo_0*"
#
# reconstruct_prefix = "rec_"

# reconstruct_format = ".tiff"
#
# focusing_distance = 0.7                    # meters
#
# # images_list = get_list_images(target_folder, holo_name_mask)
# holo = Hologram()
# holo.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/holo2.tiff")
#
# holo2 = Hologram()
# holo2.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/holo.tiff")
#
# holo.image_array = holo.image_array - holo2.image_array
# #
# # ref = Image()
# # ref.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/ref.tiff")
# #
# # obj = Image()
# # obj.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/obj.tiff")
# #
# # holo.image_array = holo.image_array - ref.image_array - obj.image_array
#
# rec = Reconstruction(holo)
# rec.filter_hologram(holo)
# roi = rec.spectrum_roi
#
# rec.propagate(focusing_distance)
#
# display_image(np.abs(rec.image_array), 0.25, "Rec")
#
#
# #
# # hologram_filtered-= np.min(hologram_filtered)
# # print(np.max((hologram_filtered)))
# # display_image(hologram_filtered, 0.5, "image")
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# holo = Hologram()
# holo.read_image_file_into_array(item)
# if "roi" not in locals():
#     rec = Reconstruction(holo)
#     rec.filter_hologram(holo)
#     roi = rec.spectrum_roi
# else:
#     rec = Reconstruction(holo, spectrum_roi=roi)
#     rec.filter_hologram(holo)
#
# rec.propagate(focusing_distance)
#
# current_file = os.path.join(results_folder, reconstruct_prefix + "{:03d}".format(itr))
# print("Copying image to file: " + current_file + reconstruct_format)
# print("... ... ...")
# rec.write_array_into_image_file(current_file, reconstruct_format)
# crop_image(current_file + reconstruct_format, current_file + reconstruct_format)


# plt.rcParams.update({'font.size': 14})
#
# target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/three"
#
# data_planar = np.load(os.path.join(target_folder, "planar_data.npz"))
# data_walsh = np.load(os.path.join(target_folder, "walsh_data.npz"))
# data_rand = np.load(os.path.join(target_folder, "random_data.npz"))
#
# sc_planar = data_planar[data_planar.files[1]]
# sc_walsh = data_walsh[data_walsh.files[1]]
# sc_rand = data_rand[data_rand.files[1]]
#
#
# #Contrast
# t = np.arange(1, len(sc_planar)+1)
# plt.plot(t, 1.0/np.sqrt(t), 'k--')
# plt.plot(t, sc_planar, 'bs')
# plt.plot(t, sc_walsh, 'go')
# plt.plot(t, sc_rand, 'rX')
# plt.xlabel('N')
# plt.xticks(t)
#
# plt.legend(('Theoretical', 'Planes', 'Walsh', "Random"),
#            bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0)
#
# plt.savefig(os.path.join(target_folder, "coeff.png"), bbox_inches="tight")
# plt.show()



# #Correlation

# target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/three"
# data_file = "data.npz"
#
# cc_speckle= data[data.files[0]]
# data = np.load(os.path.join(target_folder, data_file))

# mask =  1-np.tri(cc_speckle.shape[0], k=0)
# cc_speckle = np.ma.array(cc_speckle, mask=mask) # mask out the lower triangle
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# cmap = CM.get_cmap('viridis', 256) # jet doesn't have white color
# cmap.set_bad('w') # default value is 'k'
# ax1.imshow(cc_speckle, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
# # ax1.grid(True)
# plt.xticks(range(0,cc_speckle.shape[0], 1))
# plt.yticks(range(0,cc_speckle.shape[0], 1))
# plt.savefig(os.path.join(target_folder, "corr.png"), bbox_inches="tight")

print("Done...Goodbye")