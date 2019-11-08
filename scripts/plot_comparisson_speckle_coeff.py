# Plot multiple speckle coefficients in a single figure

from speck_rem import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 16})

results_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual"

# manually append a list with the full directory for each data.pkl file
target_folders = list()
target_folders.append(r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\composed_B20_G0064\rec/")
target_folders.append(r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\composed_B20_G0128\rec/")
target_folders.append(r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\composed_B20_G0512\rec/")

data_filename = "data.pkl"

format = ".svg"

fig = plt.figure(figsize=(8,6))
markers = ['gs-', 'b+-', 'r.-']
labels = ["64", "128", "512"]

for folder, mark, lab in zip(target_folders, markers, labels):
    print(folder)
    print(mark)
    print(lab)

    with open(os.path.join(folder, data_filename), "rb") as f:
        speckle_data = pickle.load(f)

    number_images = len(speckle_data["sc_avg"])
    t = np.arange(1, number_images + 1)
    # Speckle contrast
    plt.plot(t, speckle_data["sc_avg"]/max(speckle_data["sc_avg"]), mark, label=lab)
    # plt.plot(t, speckle_data["sc_std"]/max(speckle_data["sc_std"]), 'r*')

# plt.title("Speckle coefficient vs. number of images")
plt.plot(t, 1.0 / np.sqrt(t), 'k--')
plt.text(3, 0.39, "$1/\sqrt{N}$")
plt.xlabel("$N$")
plt.ylabel("$C$")

plt.legend()

#Add specific ticks to xlabel
xmin, xmax = ax.get_xlim()
custom_ticks = np.linspace(xmin, xmax, 11, dtype=int)
ax.set_xticks(custom_ticks)
custom_ticks_str = list(map(str, custom_ticks))
custom_ticks_str[0] = ""
ax.set_xticklabels(custom_ticks_str)


plt.savefig(os.path.join(results_folder, "coefficient" + format), bbox_inches="tight")

plt.show()

