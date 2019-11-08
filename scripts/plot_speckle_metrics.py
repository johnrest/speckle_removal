# Script to load the pickle data and plot it

from speck_rem import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\random_different_sized_grain\classic/"
data_filename = "data.pkl"

format = ".svg"

plt.rcParams.update({'font.size': 16})

with open(os.path.join(target_folder, data_filename), "rb") as f:
    speckle_data = pickle.load(f)

number_images = len(speckle_data["sc_avg"])

# Speckle contrast
t = np.arange(1, number_images+1, dtype=np.int32)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
plt.plot(t, 1.0/np.sqrt(t), 'k--')
plt.plot(t, speckle_data["sc_avg"]/max(speckle_data["sc_avg"]), 'bo')
# plt.plot(t, speckle_data["sc_std"]/max(speckle_data["sc_std"]), 'r*')
plt.xlabel("$N$")
plt.ylabel("$C$")
plt.text(3, 0.39, "$1/\sqrt{N}$")
# plt.title("Speckle coefficient vs. number of images")

#Add specific ticks to xlabel
xmin, xmax = ax.get_xlim()
custom_ticks = np.linspace(xmin, xmax, 11, dtype=int)
ax.set_xticks(custom_ticks)
custom_ticks_str = list(map(str, custom_ticks))
custom_ticks_str[0] = ""
ax.set_xticklabels(custom_ticks_str)

plt.savefig(os.path.join(target_folder, "coefficient" + format), bbox_inches="tight")

# Correlation coefficient matrix
cc_speckle= speckle_data["coefficient_matrix"]

mask = 1-np.tri(cc_speckle.shape[0], k=0)
cc_speckle = np.ma.array(cc_speckle, mask=mask) # mask out the lower triangle
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
cmap = CM.get_cmap('viridis', 256)                                  # jet doesn't have white color
cmap.set_bad('w') # default value is 'k'
im = plt.imshow(cc_speckle, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
# ax1.grid(True)
plt.xticks(range(0, cc_speckle.shape[0], 1))
plt.yticks(range(0, cc_speckle.shape[0], 1))
plt.xlabel("$t$")
plt.ylabel("$s$")
# plt.title("Correlation coefficients for the image pair (t,s)")
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.savefig(os.path.join(target_folder, "corr" + format), bbox_inches="tight")

plt.show()


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
