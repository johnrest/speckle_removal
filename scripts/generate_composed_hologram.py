# Script to generate composed holograms with sampling from the
# originals

from speck_rem import *

def unique_3D_random_sampling(size, number_patterns):
    """
    Creates a set of 3D sampling patterns where one sample
    is selected only once from the volume.
    :param size: size of the pattern
    :param number_patterns: number of patterns to compute
    :return: batch: list of numpy arrays with the patterns
    """
    w = size
    h = size

    ii, jj = np.meshgrid(np.linspace(0, w, w, endpoint=False),
                         np.linspace(0, h, h, endpoint=False))

    ii = ii.astype(int)
    jj = jj.astype(int)

    ii = ii.ravel()
    jj = jj.ravel()

    p = np.random.permutation(len(ii))
    ps = np.array_split(p, number_patterns)

    batch = []
    for itr, item in enumerate(range(number_patterns)):
        pattern = np.zeros((size, size))
        pattern[ii[ps[itr]], jj[[ps[itr]]]] = 1.0
        batch.append(pattern)

    return batch

target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual/"
results_folder = create_folder(target_folder, "composed_B20_G0064")
hologram_name_mask = "holo_0*"
filename_base = "comp_holo_"

images_list = get_list_images(target_folder, hologram_name_mask)

hologram = Hologram()
hologram.read_image_file_into_array(images_list[0])
w, h = hologram.image_array.shape

_, file_extension = os.path.splitext(images_list[0])

batch_size = 20
grain = 64
pattern_size = int(w/grain)
basis = len(images_list)

for itr, _ in enumerate(range(batch_size)):

    batch_length = min([basis, pattern_size**2])
    pattern_batch = unique_3D_random_sampling(pattern_size, batch_length)

    composed = Hologram()
    composed.image_array = np.zeros((w, h), dtype=np.float32)

    idx_sel = np.random.choice(len(images_list), basis, replace=False)

    for jtr, pattern in enumerate(pattern_batch):
        binary_mask = pattern.repeat(grain, axis=0).repeat(grain, axis=1)

        # binary_mask = cv2.blur(binary_mask, (20, 20))

        # Store each individual mask OPTIONAL
        mask = Image()
        mask.image_array = binary_mask
        current_file = os.path.join(results_folder, "mask_" + "{:02d}".format(itr) + "_{:02d}".format(jtr))
        mask.write_array_into_image_file(current_file, file_extension)

        hologram = Hologram()
        hologram.read_image_file_into_array(images_list[idx_sel[jtr]])
        hologram.image_array -= np.mean(hologram.image_array)
        composed.image_array = composed.image_array + hologram.image_array*binary_mask

        # Test to store all individual holograms
        # composed.image_array = hologram.image_array*binary_mask
        # current_file = os.path.join(results_folder, filename_base + "{:02d}".format(itr) +
        #                             "_{:02d}".format(jtr))
        # print('Creating...', current_file)
        # composed.write_array_into_image_file(current_file, file_extension)


    current_file = os.path.join(results_folder, filename_base + "{:03d}".format(itr))
    print('Creating...', current_file)
    composed.write_array_into_image_file(current_file, file_extension)


# cv2.destroyAllWindows()