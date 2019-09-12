# Full process with the classic/standard methodology for a folder/file
from speck_rem import *

target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual/"
results_folder = create_folder(target_folder, "classic")
hologram_name_mask = "holo_*"
reconstruct_prefix = "rec_"
reconstruct_format = ".tiff"
data_filename = "data"
roi_mode = "load"                       #load or save

focusing_distance = 0.85                    # meters

images_list = get_list_images(target_folder, hologram_name_mask)

if roi_mode == "load":
    current_file = os.path.join(results_folder, "roi01")
    roi = load_roi(current_file)

for itr, item in enumerate(images_list):
    print("Processing hologram :", item)
    print("... ... ...")

    hologram = Hologram()
    hologram.read_image_file_into_array(item)

    if itr == 0 and roi_mode == "save":
        rec = Reconstruction(hologram)
        rec.image_array = hologram.image_array
        rec.propagate(focusing_distance)
        roi = select_roi(np.sqrt(np.abs(rec.image_array)), "Select ROI")
        current_file = os.path.join(results_folder, "roi01")
        save_roi(roi, current_file)
        # rec.filter_hologram(hologram)
        # roi = rec.spectrum_roi
    else:
        rec = Reconstruction(hologram)
        rec.image_array = hologram.image_array
        rec.propagate(focusing_distance)
        # rec.filter_hologram(hologram)

    # rec.propagate(focusing_distance)
    rec.image_array = crop_array_with_roi(rec.image_array, roi)

    current_file = os.path.join(results_folder, reconstruct_prefix + "{:03d}".format(itr))
    print("Copying image to file: " + current_file + reconstruct_format)
    print("... ... ...")
    rec.write_array_into_image_file(current_file, reconstruct_format)
    # crop_image(current_file+reconstruct_format, current_file+reconstruct_format)


# List all reconstructed images
reconstruction_list = get_list_images(results_folder, reconstruct_prefix+"*")

# Create dictionary to store metrics
speckle_data = dict()

# Compute and store speckle metrics
print("Computing the speckle metrics coefficients...")

if roi_mode == "save":
    roi = select_roi(np.abs(rec.image_array), "Select ROI")
    current_file = os.path.join(results_folder, "roi02")
    save_roi(roi, current_file)
elif roi_mode == "load":
    current_file = os.path.join(results_folder, "roi02")
    roi = load_roi(current_file)

results = speckle_metrics(reconstruction_list, roi)
speckle_data.update(zip(("sc_avg", "sc_std", "ssi_avg", "ssi_std", "smpi_avg", "smpi_std"), results))

# Compute the speckle correlation coefficient matrix
print("Computing the correlation matrix...")
speckle_data["coefficient_matrix"] = speckle_correlation_coefficient(reconstruction_list, roi=False)

# Store speckle statistics in a python binary file
data_filename = os.path.join(results_folder, data_filename+".pkl")
with open(data_filename, "wb") as f:
    pickle.dump(speckle_data, f)

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