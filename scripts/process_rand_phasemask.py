# Process a folder/file with the random phase mask applied to the hologram.

from speck_rem import *

target_folder = "C:/Users/itm/Desktop/DH/2018_10_05/test"
holo_name_mask = "holo_0*"
phasemask_prefix = "phasemask_"
reconstruct_prefix = "rec_"
focusing_distance = 1.3
recon_batch = list()
reconstruct_format = ".bmp"
N = 20

images_list = get_list_images(target_folder, holo_name_mask)


for itr, item in enumerate(range(0, N)):
    print("Processing hologram :", item)
    print("... ... ...")

    holo = Hologram()
    holo.read_image_file_into_array(images_list[0])

    phasemask = RandomPhaseMask()
    phasemask.create(1280/4)
    phasemask_filename = os.path.join(target_folder, phasemask_prefix + "{:3d}".format(itr))
    phasemask.write_phase_into_image_file(phasemask_filename, reconstruct_format)

    holo.image_array = np.multiply(holo.image_array, phasemask.image_array)

    if itr == 0:
        recon = Reconstruction(holo)
    else:
        recon = Reconstruction(holo, spectrum_roi=selected_roi)

    recon.filter_hologram(holo)
    selected_roi = recon.spectrum_roi
    prop = recon
    prop.image_array = recon.propagate(focusing_distance)

    current_file = os.path.join(target_folder, reconstruct_prefix + "{:02d}".format(itr))
    print("Copying to image: " + current_file + reconstruct_format)
    prop.write_array_into_image_file(current_file, reconstruct_format)

    recon_batch.append(prop)
    print("Finished iteration: ", itr)

#Compute the sum of amplitudes as final image and compute the speckle contrast
amplitude_sum = Image()
amplitude_sum.image_array = np.abs(prop.image_array)
speckle_contrast_list = []

roi = select_roi(np.abs(amplitude_sum.image_array), "Select ROI to compute speckle contrast")

for itr, item in enumerate(recon_batch):
        amplitude_sum.image_array += np.abs(item.image_array)
        speckle_contrast_list.append(speckle_contrast_amp(amplitude_sum.image_array, roi))

print("Copying final image")
amplitude_sum.write_array_into_image_file(os.path.join(target_folder, "amplitude_sum"), reconstruct_format)

# red dashes for theoretical and blue squares for experimental
t = np.arange(1, N+1)
plt.plot(t, speckle_contrast_list/np.max(speckle_contrast_list), 'bs', t, 1.0/np.sqrt(t), 'r--')
plt.title("Blue: Exp, Red: T")


# OLD:
# sc = speckle_contrast(recon_batch)
# print("Speckle contrasts for the individual images are: ", sc)


#Compute and plot the correlation coefficient matrix
print("Computing correlation matrix")
cc_speckle = speckle_correlation_coefficient(recon_batch, roi=True)
fig, ax = plt.subplots()
im = ax.imshow(cc_speckle, origin='lower')
fig.colorbar(im)
plt.title("Correlation coefficient matrix")


plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

