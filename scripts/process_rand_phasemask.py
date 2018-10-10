# Process a folder/file with the random phase mask applied to the hologram.

## TEMP: attempt to implement a phase diffuser for single hologram reduction
images_list = get_list_images(target_folder, target_filename + "_0*")
# for itr, item in enumerate(images_list):
for itr, item in enumerate(range(0, 10)):

    holo = Hologram()
    holo.read_image_file_into_array(images_list[0])
    holo.phase_modulate()

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

cv2.waitKey(0)
cv2.destroyAllWindows()

print("SPECKLE REMOVAL PROJECT")
target_folder = "C:/Users/itm/Desktop/DH/2018_10_05/test"
target_filename = "holo"
reconstruct_prefix = "rec_"
focusing_distance = 1.3  # 1.7 for dice rotating / 1.3 for dice walsh
recon_batch = list()


