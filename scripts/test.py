# Testing script for testing the tests
from speck_rem import *
from speck_rem.holography import Image


# # ======================================================================================================================
# Test: fastNlMeansDenoisingMulti attempt to reduce the noise, thinking of temporal measurements

target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\classic/"
file_mask = "rec_*"

images_list = get_list_images(target_folder, file_mask)
# print(images_list)
img = Image()
img.read_image_file_into_array(images_list[0])

display_image(img.image_array,1, "indiv")

images = []
for item in images_list:
    img = Image()
    img.read_image_file_into_array(item)
    images.append(img.image_array.astype("uint8"))

filtered = cv2.fastNlMeansDenoisingMulti(images, 5, 11, None, 8, 11, 41)

display_image(filtered, 1, "fastNLMeansMulti")

cv2.waitKey()
cv2.destroyAllWindows()

# # ======================================================================================================================
# Test: bilateral filter on reconstucted intensities for comparisson

# target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\classic/"
# file_mask = "rec_*"
#
# images_list = get_list_images(target_folder, file_mask)
# print(images_list)
# img = Image()
# img.read_image_file_into_array(images_list[0])
#
# display_image(img.image_array, 1, "Original")
# # cv2.waitKey()
#
# print(img.image_array.shape)
#
# filtered = cv2.bilateralFilter(img.image_array, 5, 255, 255)
#
# display_image(filtered, 1, "Bilateral filter")
# cv2.waitKey()
# cv2.destroyAllWindows()




# # ======================================================================================================================
# Test: deconvolution to improve blurred images

# def resize_with_pad(image, hp=2048, wp=2048):
#
#     def get_padding_size(image):
#         h, w = image.shape
#
#         dh = hp - h
#         top = dh // 2
#         bottom = dh - top
#         dw = wp - w
#         left = dw // 2
#         right = dw - left
#
#         return top, bottom, left, right
#
#     top, bottom, left, right = get_padding_size(image)
#     BLACK = [0, 0]
#     resized_image = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
#
#     return resized_image
#
# #Read reconstruction from file
# target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\composed_B20_G0128/rec/"
# file_mask = "aver*"
#
# images_list = get_list_images(target_folder, file_mask)
# print(images_list)
# img = Image()
# img.read_image_file_into_array(images_list[0])
#
#
# IMG = resize_with_pad(img.image_array, 2048, 2048)
# display_image(IMG, 1, "Original")
# #Compute psf
# nx, ny = 2048, 2048
# x = np.linspace(-nx/2, nx/2, num=nx, endpoint=True)
# y = np.linspace(-ny/2, ny/2, num=ny, endpoint=True)
# xx, yy = np.meshgrid(x, y, sparse=True)
# W = 128
#
# psf = (np.sin(W * xx / nx) / (W * xx / nx)) *(np.sin(W * yy / ny) / (W * yy / ny))
# psf /= psf.sum()

# IMG = cv2.dft(IMG, flags=cv2.DFT_COMPLEX_OUTPUT)
# print(np.shape(IMG))
# PSF = cv2.dft(psf, flags=cv2.DFT_COMPLEX_OUTPUT)
# PSF2 = (PSF ** 2).sum(-1)
# iPSF = PSF / (PSF2)[..., np.newaxis]
# print(np.shape(iPSF))
# RES = cv2.mulSpectrums(IMG, iPSF, 0)
# res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
# # res = np.roll(res, -kh // 2, 0)
# # res = np.roll(res, -kw // 2, 1)
# display_image(np.abs(res), 1, "Deconv")

# Alternative from http://www.radio-science.net/2017/09/deconvolution-in-frequency-domain-with.html
# FFT true image
# H = np.fft.fft2(IMG)

# FFT point spread function (first column of theory matrix G)
# P = np.fft.fft2(psf)

# simulate measurement (d=Gm + \eta)
# also normalize 2d FFT
# fftshit resolves wrapping of spectral components (0..2pi to -pi..pi)
# d = (1.0/(H.shape[0]*H.shape[1]))*np.fft.fftshift(np.fft.ifft2(H*P).real)
# add noise with standard deviation of 0.1
# d = d + np.random.randn(d.shape[0],d.shape[1])*0.1

# D = np.fft.fft2(d)

# regularization parameter
# (should be one to two orders of magnitude below the largest spectral component of point-spread function)
# alpha = 0.0000000001

# -dampped spectral components,
# -also known as Wiener filtering
# (conj(S)/(|S|^2 + alpha^2)) U^H d
# M = (np.conj(P)/(np.abs(P)**2.0 + alpha**2.0))*D

# maximum a posteriori estimate of deconvolved image
# m_map = U (conj(S)/(|S|^2 + alpha^2)) U^H d
# m_map=(H.shape[1]*H.shape[0])*np.fft.fftshift(np.fft.ifft2(M).real)

# display_image(m_map, 1.0, "Deconvolved")


# cv2.waitKey(0)
# plt.show()
#Deconvolved with image read from file


# # ======================================================================================================================

"""
# # ======================================================================================================================
## Test
target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual/"
file_mask = "holo_*"

focusing_distance = .85

images_list = get_list_images(target_folder, file_mask)

hologram = Hologram()
hologram.read_image_file_into_array(images_list[0])

mask = np.zeros_like(hologram.image_array)
mask[512:1536, 512:1536] = 1
# mask = 1-mask
display_image(mask, 0.5, "mask")

rec = Reconstruction(hologram)
rec.image_array = mask * hologram.image_array
rec.propagate(focusing_distance)
roi = select_roi(np.sqrt(np.abs(rec.image_array)), "Select ROI")

rec.image_array = crop_array_with_roi(rec.image_array, roi)

display_image(np.abs(rec.image_array), 1, "Reconstruction")

cv2.waitKey(0)

# # ======================================================================================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def doPCA(data, components=1):
    pca = PCA(components)
    pca.fit(data)
    components = pca.transform(data)
    approximation = pca.inverse_transform(components)
    return approximation

# PCA analysis of reconstructed images
target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\temp_08_21/"
file_mask = "comp_holo_*"

images_list = get_list_images(target_folder, file_mask)
images_list = images_list[0:4]
print(*images_list, sep="\n")

# Evaluate only once to crop images by a centered half
# for itr, item in enumerate(images_list):
#     crop_image(item, item)

w, h = 2048, 2048

data = np.zeros((w*h, len(images_list)))

for itr, item in enumerate(images_list):

    img = Image()
    img.read_image_file_into_array(item)
    data[:, itr] = img.image_array.reshape(-1)

# scaler = StandardScaler()

# Fit on training set only.
# data = scaler.fit_transform(data)

pca = PCA(0.5)

pca_data = pca.fit_transform(data)

print("Number of components:", pca.n_components_)

approximation = pca.inverse_transform(pca_data)

average = Image()
average.image_array = np.zeros((2048,2048))

for itr in range(len(images_list)):

    ind = Image()
    ind.image_array = approximation[:, itr].reshape(2048, 2048)
    current_file = os.path.join(target_folder, "ind_" + "{:03d}".format(itr))
    ind.write_array_into_image_file(current_file, ".tiff")
    average.image_array += ind.image_array


current_file = os.path.join(target_folder, "avg_")
average.image_array /= len(images_list)
average.write_array_into_image_file(current_file, ".tiff")


# plt.figure(figsize=(8,4))
# Original Image
# plt.subplot(1, 2, 1)
# plt.imshow(data[:,1].reshape(512,512),
#               cmap = plt.cm.gray, interpolation='nearest',
#               clim=(-1, 1));
# plt.xlabel('784 components', fontsize = 14)
# plt.title('Original Image', fontsize = 20)

# 154 principal components
# plt.subplot(1, 2, 2)
# plt.imshow(approximation[:,1].reshape(512, 512),
#               cmap = plt.cm.gray, interpolation='nearest',
#               clim=(-1, 1));
# plt.xlabel('154 components', fontsize = 14)
# plt.title('100% of Explained Variance', fontsize = 20)

#Plot the cumulative variance
# tot = sum(pca.explained_variance_)
# var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
#
# plt.figure(figsize=(10, 5))
# plt.step(range(1, pca.n_components_+1), cum_var_exp, where='mid',label='cumulative explained variance')
# plt.title('Cumulative Explained Variance as a Function of the Number of Components')
# plt.ylabel('Cumulative Explained variance')
# plt.xlabel('Principal components')

# display_image(data[:,0].reshape(512,512), 0.5, "image")
# display_image(approximation[:,0].reshape(512,512), 0.5, "approx")

# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.show()
"""
# # ======================================================================================================================
#
# def test_random_pattern(scale, number_patterns):
#     w = scale
#     h = scale
#
#     ii, jj = np.meshgrid(np.linspace(0, w, w, endpoint=False),
#                          np.linspace(0, h, h, endpoint=False))
#
#     ii = ii.astype(int)
#     jj = jj.astype(int)
#
#     ii = ii.ravel()
#     jj = jj.ravel()
#
#     p = np.random.permutation(len(ii))
#     ps = np.array_split(p, number_patterns)
#
#     batch = []
#     for itr, item in enumerate(range(number_patterns)):
#         pattern = np.zeros((scale, scale))
#         pattern[ii[ps[itr]], jj[[ps[itr]]]] = 1.0
#         batch.append(pattern)
#
#     return batch
#
#
# ### Compute a 3D composed hologram from the batch, with varying grain size
# target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/random_different_sized_grain/"
# hologram_name_mask = "holo_0*"
# basis_length = 4
# composed_hologram_name_prefix = "composed_holo_"
# file_format = ".tiff"
#
# images_list = get_list_images(target_folder, hologram_name_mask)
#
# grain = 256
# pattern_size = int(2048/grain)
# number_pattern_images = basis_length
#
# for itr, _ in enumerate(range(20)):
#
#     # pattern_batch = compute_pattern_batch(scale=pattern_size, batch_length=number_pattern_images)
#     pattern_batch = test_random_pattern(pattern_size, number_pattern_images)
#
#     composed = Hologram()
#     composed.image_array = np.zeros((2048, 2048), dtype=np.float32)
#
#     for jtr, pattern in enumerate(pattern_batch):
#         binary_mask = pattern.repeat(grain, axis=0).repeat(grain, axis=1)
#
#         hologram = Hologram()
#         hologram.read_image_file_into_array(images_list[ np.random.randint(basis_length)])
#         composed.image_array = composed.image_array + hologram.image_array*binary_mask
#         # display_image(composed.image_array, 0.4, "Partial")
#         # cv2.waitKey(0)
#
#     current_file = os.path.join(target_folder, composed_hologram_name_prefix + "{:03d}".format(itr))
#     print('Creating...', current_file)
#     composed.write_array_into_image_file(current_file, file_format)
#
# display_image(composed.image_array)

# rec = Reconstruction(composed)
# rec.filter_hologram(composed)
#
# rec.propagate(0.85)
#
# display_image(np.abs(rec.image_array), 0.5, "rec")
#
# cv2.waitKey()
# print("Done...")

# ======================================================================================================================
"""
### Compute the 3D random resampling of the hologram batch
target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/random_different_sized_grain/"
hologram_name_mask = "holo_0*"
basis_length = 4
composed_hologram_name_prefix = "holo_"
file_format = ".tiff"

images_list = get_list_images(target_folder, hologram_name_mask)

h1 = Hologram()
h1.read_image_file_into_array(images_list[0])

w, h = h1.image_array.shape

ii, jj = np.meshgrid(np.linspace(0, w, w, endpoint=False),
                   np.linspace(0, h, h, endpoint=False) )

ii = ii.astype(int)
jj = jj.astype(int)

ii = ii.ravel()
jj = jj.ravel()


for itr, _ in enumerate(range(len(images_list) - basis_length)):
    sub_images_list = images_list[itr: itr+basis_length]

    resampled_hologram = Hologram()
    # resampled_hologram.read_image_file_into_array(images_list[0])
    resampled_hologram.image_array = np.zeros((w,h))

    p = np.random.permutation(len(ii))
    ps = np.array_split(p, len(sub_images_list))

    for jtr, item in enumerate(sub_images_list):
        hologram = Hologram()
        hologram.read_image_file_into_array(item)
        resampled_hologram.image_array[ii[ps[jtr]], jj[[ps[jtr]]]] = hologram.image_array[ii[ps[jtr]], jj[[ps[jtr]]]]

    current_file = os.path.join(target_folder, composed_hologram_name_prefix + "{:03d}".format(itr+20))
    resampled_hologram.write_array_into_image_file(current_file, file_format)
"""
# ======================================================================================================================

# display_image(resampled_hologram.image_array)
#
# rec = Reconstruction(resampled_hologram)
# rec.filter_hologram(resampled_hologram)
#
# rec.propagate(0.85)
#
# display_image(np.abs(rec.image_array), 1, "rec")
#
# cv2.waitKey()



# binary_mask = np.random.randint(2, size=( w, h))
#
# h2 = Hologram()
# h2.read_image_file_into_array(images_list[1])
#
# nh = Hologram()
# # nh.image_array = h1.image_array*binary_mask + h2.image_array*(1-binary_mask)
#
#
# ii, jj = np.meshgrid(np.linspace(0, w, w, endpoint=False),
#                    np.linspace(0, h, h, endpoint=False) )
#
# ii = ii.astype(int)
# jj = jj.astype(int)
#
#
# ind = np.arange(w*h)
# np.random.shuffle(ind)
# ii = ii.ravel()
# jj = jj.ravel()
#
# p = np.random.permutation(len(ii))
#
# ii = ii[p]
# jj = jj[p]
#
# # ii = ii.reshape((w,h))
# # jj = jj.reshape((w,h))
#
# h1.image_array[ii[0:2000000], jj[0:2000000]] = 0.0
#
# display_image(h1.image_array)
#
# #
# # rec = Reconstruction(nh)
# # rec.filter_hologram(nh)
# #
# # rec.propagate(0.85)
# #
# # display_image(binary_mask, 1, "rec")
#
#


#
# # ii = ii.reshape((1,9))
# # jj = ii.reshape((1,9))
#
# ii = ii.ravel()
# jj = jj.ravel()
#
#
#
# # # print(ii)
# # np.random.shuffle(ii)
# # ii = ii.reshape((3,3))
#
# array = np.eye(3,3)
# # print(ii)
# array[ii,jj] = 0.5
# print(array)



# ======================================================================================================================
"""Compute the difference between two holograms and save to file"""

# target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/random_different_sized_grain/"
# hologram_name_mask = "holo_0*"
# images_list = get_list_images(target_folder, hologram_name_mask)
#
# for itr, item in enumerate(images_list[1:]):
#     print("Processing hologram :", item)
#     print("... ... ...")
#
#     hologram = Hologram()
#     hologram.read_image_file_into_array(item)
#
#     holo_sub = Hologram()
#     holo_sub.read_image_file_into_array(images_list[itr-1])
#
#     hologram.image_array = hologram.image_array - holo_sub.image_array
#     hologram.image_array -= np.min(hologram.image_array)
#
#     current_file = os.path.join(target_folder, "holo_" + "{:03d}".format(itr+20))
#
#     hologram.write_array_into_image_file(current_file, ".tiff")
# ======================================================================================================================

# # target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/three/planar/"
# target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/planar_fixed_freq_manual/"
# results_folder = create_folder(target_folder, "comp")
# holo_name_mask = "holo_*"
# reconstruct_prefix = "full_recon"
# results_folder = target_folder
# reconstruct_format = ".tiff"
#
#
# focusing_distance = 0.85                    # meters
# # focusing_distance = 1.5                    # meters
#
# images_list = get_list_images(target_folder, holo_name_mask)
#
# holo_total = np.zeros((2048, 2048), dtype=np.float32)
# # holo_total.read_image_file_into_array(images_list[0])
#
# for itr, item in enumerate(images_list):
#     holo = Hologram()
#     holo.read_image_file_into_array(item)
#     holo_total += holo.image_array
#
# holo.image_array = holo_total
# rec = Reconstruction(holo)
# rec.filter_hologram(holo)
# roi = rec.spectrum_roi
#
# rec.propagate(focusing_distance)
#
#
# display_image(holo.image_array, 0.5, "holo")
# display_image(np.abs(rec.image_array), 0.5, "Rec")
#
# current_file = os.path.join(results_folder, reconstruct_prefix)
# print("Copying image to file: " + current_file + reconstruct_format)
# print("... ... ...")
# rec.write_array_into_image_file(current_file, reconstruct_format)
# crop_image(current_file + reconstruct_format, current_file + reconstruct_format)
#
# cv2.waitKey()

# target_folder = "C:/Users/itm/Desktop/DH/2018_11_22/three/random_different_sized_grain/"
# results_folder = create_folder(target_folder, "comp")
# holo_name_mask = "holo_0*"
#
# reconstruct_prefix = "rec_"

# reconstruct_format = ".tiff"
#
# focusing_distance = 0.7                    # meters
#
# # images_list = get_list_images(target_folder, holo_name_mask)
# holo = Hologram()
# holo.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/holo2.tiff")
#
# holo2 = Hologram()
# holo2.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/holo.tiff")
#
# holo.image_array = holo.image_array - holo2.image_array
# #
# # ref = Image()
# # ref.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/ref.tiff")
# #
# # obj = Image()
# # obj.read_image_file_into_array("C:/Users/itm/Desktop/DH/2018_12_07/test/obj.tiff")
# #
# # holo.image_array = holo.image_array - ref.image_array - obj.image_array
#
# rec = Reconstruction(holo)
# rec.filter_hologram(holo)
# roi = rec.spectrum_roi
#
# rec.propagate(focusing_distance)
#
# display_image(np.abs(rec.image_array), 0.25, "Rec")
#
#
# #
# # hologram_filtered-= np.min(hologram_filtered)
# # print(np.max((hologram_filtered)))
# # display_image(hologram_filtered, 0.5, "image")
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# holo = Hologram()
# holo.read_image_file_into_array(item)
# if "roi" not in locals():
#     rec = Reconstruction(holo)
#     rec.filter_hologram(holo)
#     roi = rec.spectrum_roi
# else:
#     rec = Reconstruction(holo, spectrum_roi=roi)
#     rec.filter_hologram(holo)
#
# rec.propagate(focusing_distance)
#
# current_file = os.path.join(results_folder, reconstruct_prefix + "{:03d}".format(itr))
# print("Copying image to file: " + current_file + reconstruct_format)
# print("... ... ...")
# rec.write_array_into_image_file(current_file, reconstruct_format)
# crop_image(current_file + reconstruct_format, current_file + reconstruct_format)


# plt.rcParams.update({'font.size': 14})
#
# target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/three"
#
# data_planar = np.load(os.path.join(target_folder, "planar_data.npz"))
# data_walsh = np.load(os.path.join(target_folder, "walsh_data.npz"))
# data_rand = np.load(os.path.join(target_folder, "random_data.npz"))
#
# sc_planar = data_planar[data_planar.files[1]]
# sc_walsh = data_walsh[data_walsh.files[1]]
# sc_rand = data_rand[data_rand.files[1]]
#
#
# #Contrast
# t = np.arange(1, len(sc_planar)+1)
# plt.plot(t, 1.0/np.sqrt(t), 'k--')
# plt.plot(t, sc_planar, 'bs')
# plt.plot(t, sc_walsh, 'go')
# plt.plot(t, sc_rand, 'rX')
# plt.xlabel('N')
# plt.xticks(t)
#
# plt.legend(('Theoretical', 'Planes', 'Walsh', "Random"),
#            bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0)
#
# plt.savefig(os.path.join(target_folder, "coeff.png"), bbox_inches="tight")
# plt.show()



# #Correlation

# target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/three"
# data_file = "data.npz"
#
# cc_speckle= data[data.files[0]]
# data = np.load(os.path.join(target_folder, data_file))

# mask =  1-np.tri(cc_speckle.shape[0], k=0)
# cc_speckle = np.ma.array(cc_speckle, mask=mask) # mask out the lower triangle
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# cmap = CM.get_cmap('viridis', 256) # jet doesn't have white color
# cmap.set_bad('w') # default value is 'k'
# ax1.imshow(cc_speckle, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
# # ax1.grid(True)
# plt.xticks(range(0,cc_speckle.shape[0], 1))
# plt.yticks(range(0,cc_speckle.shape[0], 1))
# plt.savefig(os.path.join(target_folder, "corr.png"), bbox_inches="tight")

print("Done...Goodbye")