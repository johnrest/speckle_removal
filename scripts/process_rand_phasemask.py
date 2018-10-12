# Process a folder/file with the random phase mask applied to the hologram.

from speck_rem import *

target_folder = "C:/Users/itm/Desktop/DH/2018_10_05/test"
holo_name_mask = "holo_0*"
phasemask_prefix = "phasemask_"
reconstruct_prefix = "rec_"
focusing_distance = 1.3
recon_batch = list()
reconstruct_format = ".bmp"

images_list = get_list_images(target_folder, holo_name_mask)

for itr, item in enumerate(range(0, 10)):
    print("Processing hologram :", item)
    print("... ... ...")

    holo = Hologram()
    holo.read_image_file_into_array(images_list[0])

    phasemask = RandomPhaseMask()
    phasemask.create(1280/4)
    phasemask_filename = os.path.join(target_folder, phasemask_prefix + "{:3d}".format(itr))
    phasemask.write_array_into_image_file(phasemask_filename, reconstruct_format)

    holo.image_array = np.multiply(holo.image_array, phasemask.image_array)

    if itr == 0:
        recon = Reconstruction(holo)
    else:
        recon = Reconstruction(holo, spectrum_roi=selected_roi)

    recon.filter_hologram(holo)
    selected_roi = recon.spectrum_roi
    prop = recon
    prop.image_array = recon.propagate(focusing_distance)

    # display_image(abs(prop.image_array), 0.5, "Propagated amplitude " + "{:2d}".format(itr))
    recon_batch.append(prop)
    print("Finished iteration: ", itr)

average = np.zeros(holo.image_array.shape)
for img in recon_batch:
    average = np.add(average, abs(img.image_array))

display_image(average, 0.5, "sum")

# final = recon_batch[0]
# final.image_array = average
# recon_batch.append(final)

sc = speckle_contrast(recon_batch)
print("Speckle contrasts for the individual images are: ", sc)



cv2.waitKey(0)
cv2.destroyAllWindows()

