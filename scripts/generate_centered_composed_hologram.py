# Script to generate composed holograms with centered rectangular frames sampling from the
# original holograms

from speck_rem import *

target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual/"
results_folder = create_folder(target_folder, "temp_27_08")
hologram_name_mask = "holo_0*"
filename_base = "comp_holo_"

images_list = get_list_images(target_folder, hologram_name_mask)

hologram = Hologram()
hologram.read_image_file_into_array(images_list[0])
w, h = hologram.image_array.shape

_, file_extension = os.path.splitext(images_list[0])

wb, hb = 1024, 1024  # with of center block

frames = 1  # number of frames around the center block
basis = 20  # number of holograms to sample from
batch_size = 20  # Number of composed holograms to generate

px, py = 0, 0
dx = (w / 2 - wb / 2) / frames
dy = (h / 2 - hb / 2) / frames

for jtr, item in enumerate(range(batch_size)):

    idx_sel = np.random.choice(len(images_list), basis, replace=False)
    composed = Hologram()
    composed.image_array = np.zeros((w, h), dtype=np.float32)

    mask = np.zeros((w, h))
    added_mask = np.zeros((w, h))

    for itr, _ in enumerate(range(frames + 1)):

        if itr == 0:
            px = w / 2 - wb / 2
            py = h / 2 - hb / 2
            mask[int(py):int(py + hb), int(px):int(px + wb)] = 1
            added_mask += mask
        else:
            px = px - dx
            py = py - dy
            mask[int(py):int(py + hb + 2 * dy * itr), int(px):int(px + wb + 2 * dx * itr)] = 1
            mask = mask - added_mask
            added_mask += mask

        hologram = Hologram()
        hologram.read_image_file_into_array(images_list[idx_sel[itr]])
        hologram.image_array -= np.mean(hologram.image_array)
        composed.image_array += hologram.image_array*mask

    current_file = os.path.join(results_folder, filename_base + "{:03d}".format(jtr))
    print('Creating...', current_file)
    composed.write_array_into_image_file(current_file, file_extension)

    # display_image(composed.image_array, 0.5, "composed")
    # cv2.waitKey(0)


print("Done...")
