# Full process with the classic/standard methodology for a folder/file

from speck_rem import *

target_folder = "C:/Users/itm/Desktop/DH/2018_11_01/test_walsh"
holo_name_mask = "holo__0*"
reconstruct_prefix = "rec_"
focusing_distance = 1.3
recon_batch = list()
reconstruct_format = ".tiff"

images_list = get_list_images(target_folder, holo_name_mask)

for itr, item in enumerate(images_list):
    print("Processing hologram :", item)
    print("... ... ...")

    holo = Hologram()
    holo.read_image_file_into_array(item)

    if itr == 0:                                #select ROI for first hologram
        recon = Reconstruction(holo)
    else:                                       #Use selected ROI for the rest of holograms
        recon = Reconstruction(holo, spectrum_roi=roi)

    recon.filter_hologram(holo)
    roi = recon.spectrum_roi
    prop = recon
    prop.image_array = recon.propagate(focusing_distance)
    recon_batch.append(prop)

    display_image(abs(prop.image_array), 0.5, "Propagated amplitude")

    current_file = os.path.join(target_folder, reconstruct_prefix + "{:02d}".format(itr))
    print("Copying to image: " + current_file + reconstruct_format)
    prop.write_array_into_image_file(current_file, reconstruct_format)
    crop_image(current_file+reconstruct_format, current_file+reconstruct_format)


#Compute the sum of amplitudes as final image and compute the speckle contrast
amplitude_sum = Image()
amplitude_sum.image_array = np.abs(prop.image_array)
speckle_contrast_list = []

roi = select_roi(np.abs(amplitude_sum.image_array), "Select ROI to compute speckle contrast")

for itr, item in enumerate(recon_batch):
        amplitude_sum.image_array += np.abs(item.image_array)
        speckle_contrast_list.append(speckle_contrast_amp(amplitude_sum.image_array, roi))

print("Copying final image")
amplitude_sum_file = os.path.join(target_folder, "amplitude_sum")
amplitude_sum.write_array_into_image_file(amplitude_sum_file, reconstruct_format)
crop_image(amplitude_sum_file+reconstruct_format,amplitude_sum_file+reconstruct_format)

# red dashes for theoretical and blue squares for experimental
t = np.arange(1, len(images_list)+1)
plt.plot(t, speckle_contrast_list/np.max(speckle_contrast_list), 'bs', t, 1.0/np.sqrt(t), 'r--')
plt.title("Blue: Exp, Red: Theor")
plt.savefig(os.path.join(target_folder, "coeff.png"), bbox_inches="tight")
plt.xlabel('N'), plt.xlabel('A.U.')


#Compute and plot the correlation coefficient matrix
print("Computing correlation matrix")
cc_speckle = speckle_correlation_coefficient(recon_batch, roi=True)
fig, ax = plt.subplots()
im = ax.imshow(cc_speckle, origin='lower')
fig.colorbar(im)
plt.title("Correlation coefficient matrix")
plt.savefig(os.path.join(target_folder, "corr.png"), bbox_inches="tight")

#OLD: Compute the speckle contrast
# sc = speckle_contrast(recon_batch)
# print("Speckle contrast is: ", sc)
# plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
