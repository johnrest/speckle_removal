# Process holograms combining single and multiple holograms techniques

from speck_rem import *


target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/random_different_sized_grain/"
results_folder = create_folder(target_folder, "combined_B")
holo_name_mask = "holo_0*"
reconstruct_prefix = "rec_"
focusing_distance = 0.85
recon_batch = list()
reconstruct_format = ".tiff"
data_filename = "data"

# Parameters for SH speckle removal
phasemask_prefix = "phasemask_"
grain = 256
pattern_size = int(2048/grain)
number_pattern_images = (pattern_size*pattern_size)/4

images_list = get_list_images(target_folder, holo_name_mask)

for itr, file in enumerate(images_list):
    print("Processing hologram :", file)
    print("... ... ...")

    #Compute FCN sampled masks
    pattern_batch = compute_pattern_batch(scale=pattern_size, batch_length=number_pattern_images)

    holo = Hologram()
    holo.read_image_file_into_array(images_list[itr])

    if itr == 0:
        recon = Reconstruction(holo)
    else:
        recon = Reconstruction(holo, spectrum_roi=selected_roi)

    individual_recon = Image()
    individual_recon.image_array = np.zeros(holo.image_array.shape)

    for jtr, pattern in enumerate(pattern_batch):
        print("Processing phase mask:", jtr)
        print("... ... ...")

        phasemask = FairnessConstraintMask()
        phasemask.compute(grain, pattern)

        # phasemask_filename = os.path.join(results_folder, phasemask_prefix + "{:3d}".format(itr))
        # phasemask.write_phase_into_image_file(phasemask_filename, reconstruct_format)

        holo.image_array = np.multiply(holo.image_array, phasemask.image_array)

        recon.filter_hologram(holo)
        selected_roi = recon.spectrum_roi
        prop = recon
        prop.image_array = recon.propagate(focusing_distance)

        individual_recon.image_array += np.abs(prop.image_array)
        print("Finished iteration: ", itr)

    recon_batch.append(individual_recon)

    display_image(abs(individual_recon.image_array), 0.5, "Propagated amplitude after SH speckle removal")

    current_file = os.path.join(results_folder, reconstruct_prefix + "{:02d}".format(itr))
    print("Copying to image: " + current_file + reconstruct_format)
    individual_recon.write_array_into_image_file(current_file, reconstruct_format)
    crop_image(current_file+reconstruct_format, current_file+reconstruct_format)

#Compute the sum of amplitudes as final image and compute the speckle contrast
amplitude_sum = Image()
amplitude_sum.image_array = np.abs(individual_recon.image_array)
speckle_contrast_list = []

roi = select_roi(np.abs(amplitude_sum.image_array), "Select ROI to compute speckle contrast")

for itr, item in enumerate(recon_batch):
        amplitude_sum.image_array += np.abs(item.image_array)
        speckle_contrast_list.append(speckle_contrast_amp(amplitude_sum.image_array, roi))

print("Copying final image")
amplitude_sum_file = os.path.join(results_folder, "amplitude_sum")
amplitude_sum.write_array_into_image_file(amplitude_sum_file, reconstruct_format)
crop_image(amplitude_sum_file+reconstruct_format, amplitude_sum_file+reconstruct_format)

# red dashes for theoretical and blue squares for experimental
t = np.arange(1, len(images_list)+1)
plt.plot(t, speckle_contrast_list/np.max(speckle_contrast_list), 'bs', t, 1.0/np.sqrt(t), 'r--')
plt.title("Blue: Exp, Red: Theor")
plt.xlabel('N'), plt.xlabel('A.U.')
plt.savefig(os.path.join(results_folder, "coeff.png"), bbox_inches="tight")

#Compute and plot the correlation coefficient matrix
print("Computing correlation matrix")
cc_speckle = speckle_correlation_coefficient(recon_batch, roi=True)
fig, ax = plt.subplots()
im = ax.imshow(cc_speckle, origin='lower')
fig.colorbar(im)
plt.title("Correlation coefficient matrix")
plt.savefig(os.path.join(results_folder, "corr.png"), bbox_inches="tight")


print("Finished ... Goodbye")

# cv2.waitKey(0)
cv2.destroyAllWindows()

#TODO: update script to new functionality