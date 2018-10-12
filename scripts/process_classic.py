# Full process with the classic/standard methodology for a folder/file

from speck_rem import *

target_folder = "C:/Users/itm/Desktop/DH/2018_10_05/test"
holo_name_mask = "holo_0*"
reconstruct_prefix = "rec_"
focusing_distance = 1.3
recon_batch = list()
reconstruct_format = ".bmp"

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

# speckle_correlation_coefficient(recon_batch, roi=True)

sc = speckle_contrast(recon_batch)
print("Speckle contrast is: ", sc)

cv2.waitKey(0)
cv2.destroyAllWindows()
