# Full process with the classic/standard methodology for a folder/file
from speck_rem import *

target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/random_different_sized_grain/"
results_folder = create_folder(target_folder, "comp")
hologram_name_mask = "holo_0*"
reconstruct_prefix = "rec_"
reconstruct_format = ".tiff"
data_filename = "data"

focusing_distance = 0.85                    # meters

images_list = get_list_images(target_folder, hologram_name_mask)
# images_list = images_list[1:]

for itr, item in enumerate(images_list):
    print("Processing hologram :", item)
    print("... ... ...")

    hologram = Hologram()
    hologram.read_image_file_into_array(item)

    # holo_sub = Hologram()
    # holo_sub.read_image_file_into_array(images_list[itr-1])
    #
    # hologram.image_array = hologram.image_array - holo_sub.image_array
    # hologram.image_array -= np.min(hologram.image_array)

    if "roi" not in locals():
        rec = Reconstruction(hologram)
        rec.filter_hologram(hologram)
        roi = rec.spectrum_roi
    else:
        rec = Reconstruction(hologram, spectrum_roi=roi)
        rec.filter_hologram(hologram)

    rec.propagate(focusing_distance)

    current_file = os.path.join(results_folder, reconstruct_prefix + "{:03d}".format(itr))
    print("Copying image to file: " + current_file + reconstruct_format)
    print("... ... ...")
    rec.write_array_into_image_file(current_file, reconstruct_format)
    crop_image(current_file+reconstruct_format, current_file+reconstruct_format)

# List all reconstructed images
reconstruction_list = get_list_images(results_folder, reconstruct_prefix+"*")

# Create dictionary to store metrics
speckle_data = dict()

# Compute and store speckle metrics
print("Computing the speckle metrics coefficients...")
results = speckle_metrics(reconstruction_list)
speckle_data.update(zip(("sc_avg", "sc_std", "ssi_avg", "ssi_std", "smpi_avg", "smpi_std"), results))

# Compute the speckle correlation coefficient matrix
print("Computing the correlation matrix...")
speckle_data["coefficient_matrix"] = speckle_correlation_coefficient(reconstruction_list, roi=True)

# Store speckle statistics in a python binary file
data_filename = os.path.join(results_folder, data_filename+".pkl")
with open(data_filename, "wb") as f:
    pickle.dump(speckle_data, f)

# TODO: write scripts for plotting

# Compute standard deviation image
print("Computing the standard deviation image...")
current_file = os.path.join(results_folder, "standard_deviation")
superposition_standard_dev(reconstruction_list, current_file, reconstruct_format)

# Compute average image
print("Computing the average image...")
current_file = os.path.join(results_folder, "average")
superposition_average(reconstruction_list, current_file, reconstruct_format)

print("Finished.")
print("==============================================================================================================")

"""
print("Copying final image")
amplitude_sum_file = os.path.join(target_folder, "amplitude_sum")
amplitude_sum.write_array_into_image_file(amplitude_sum_file, reconstruct_format)
crop_image(amplitude_sum_file+reconstruct_format,amplitude_sum_file+reconstruct_format)

speckle_contrast_list = speckle_contrast_list/np.max(speckle_contrast_list)
t = np.arange(1, len(images_list)+1)
plt.plot(t, speckle_contrast_list/np.max(speckle_contrast_list), 'bs', t, 1.0/np.sqrt(t), 'r--')
plt.title("Blue: Exp, Red: Theor")
plt.xlabel('N'), plt.xlabel('A.U.')
plt.savefig(os.path.join(target_folder, "coeff.png"), bbox_inches="tight")

if itr == 0:
    amplitude_sum.image_array = np.abs(prop.image_array)
    roi_speckle = select_roi(np.abs(prop.image_array), "Select ROI to compute speckle contrast")
else:
    amplitude_sum.image_array += np.abs(prop.image_array)

speckle_contrast_list.append(speckle_contrast_amp(amplitude_sum.image_array, roi_speckle))

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
speckle_contrast_list = speckle_contrast_list/np.max(speckle_contrast_list)
t = np.arange(1, len(images_list)+1)
plt.plot(t, speckle_contrast_list/np.max(speckle_contrast_list), 'bs', t, 1.0/np.sqrt(t), 'r--')
plt.title("Blue: Exp, Red: Theor")
plt.xlabel('N'), plt.xlabel('A.U.')
plt.savefig(os.path.join(target_folder, "coeff.png"), bbox_inches="tight")

#Compute and plot the correlation coefficient matrix
print("Computing correlation matrix")
cc_speckle = speckle_correlation_coefficient(recon_batch, roi=True)
fig, ax = plt.subplots()
im = ax.imshow(cc_speckle, origin='lower')
fig.colorbar(im)
plt.title("Correlation coefficient matrix")
plt.savefig(os.path.join(target_folder, "corr.png"), bbox_inches="tight")


##Store speckle computations on arrays for future plotting
data_file = os.path.join(target_folder, "data")
np.savez(data_file, cc_speckle, speckle_contrast_list)

# # Compute and plot the correlation coefficient matrix
# mask =  np.tri(cc_speckle.shape[0], k=-1)
# cc_speckle = np.ma.array(cc_speckle, mask=mask) # mask out the lower triangle
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# cmap = CM.get_cmap('jet', 10) # jet doesn't have white color
# cmap.set_bad('w') # default value is 'k'
# ax1.imshow(cc_speckle, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
# ax1.grid(True)
# plt.show()


"""
