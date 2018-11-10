# Process a folder/file with a fcn phase mask applied to the hologram.

from speck_rem import *

target_folder = "C:/Users/itm/Desktop/DH/2018_11_09/three/planar_fixed_freq"
results_folder = "C:/Users/itm/Desktop/DH/2018_11_09/three/planar_fixed_freq/G040N016"
holo_name_mask = "holo__0*"
phasemask_prefix = "phasemask_"
reconstruct_prefix = "rec_"
focusing_distance = 1.3
recon_batch = list()
reconstruct_format = ".tiff"

# Get holograms on target folder
images_list = get_list_images(target_folder, holo_name_mask)

#Compute pattern batch with the FC rule
grain = 80
pattern_size = int(1280/grain)
number_pattern_images = (pattern_size*pattern_size)/16

pattern_batch = compute_pattern_batch(scale=pattern_size, batch_length=number_pattern_images)

for itr, item in enumerate(pattern_batch):
    print("Processing phase mask:", itr)
    print("... ... ...")

    holo = Hologram()
    holo.read_image_file_into_array(images_list[0])

    phasemask = FairnessConstraintMask()
    phasemask.compute(grain, item)

    phasemask_filename = os.path.join(results_folder, phasemask_prefix + "{:3d}".format(itr))
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

    current_file = os.path.join(results_folder, reconstruct_prefix + "{:02d}".format(itr))
    print("Copying to image: " + current_file + reconstruct_format)
    prop.write_array_into_image_file(current_file, reconstruct_format)
    crop_image(current_file + reconstruct_format, current_file + reconstruct_format)

    recon_batch.append(prop)
    print("Finished iteration: ", itr)

#Compute the sum of amplitudes as final image and compute the speckle contrast
amplitude_sum = Image()
amplitude_sum.image_array = np.zeros(prop.image_array.shape)
speckle_contrast_list = []

roi = select_roi(np.abs(prop.image_array), "Select ROI to compute speckle contrast")

for itr, item in enumerate(recon_batch):
        amplitude_sum.image_array += np.abs(item.image_array)
        speckle_contrast_list.append(speckle_contrast_amp(amplitude_sum.image_array, roi))

print("Saving to file average image")
amplitude_sum_file = os.path.join(results_folder, "amplitude_sum")
amplitude_sum.write_array_into_image_file(amplitude_sum_file, reconstruct_format)
crop_image(amplitude_sum_file+reconstruct_format,amplitude_sum_file+reconstruct_format)

# red dashes for theoretical and blue squares for experimental
t = np.arange(1, len(pattern_batch)+1)
plt.plot(t, speckle_contrast_list/np.max(speckle_contrast_list), 'bs', t, 1.0/np.sqrt(t), 'r--')
plt.title("Blue: Exp, Red: T")
plt.xlabel('N'), plt.ylabel('A.U.')
plt.savefig(os.path.join(results_folder, "coeff.png"), bbox_inches="tight")


#Compute and plot the correlation coefficient matrix
print("Computing correlation matrix")
cc_speckle = speckle_correlation_coefficient(recon_batch, roi=True)
fig, ax = plt.subplots()
im = ax.imshow(cc_speckle, origin='lower')
fig.colorbar(im)
plt.title("Correlation coefficient matrix")
plt.savefig(os.path.join(results_folder, "corr.png"), bbox_inches="tight")

#cv2.waitKey(0)
cv2.destroyAllWindows()

print("Finished...goodbye")