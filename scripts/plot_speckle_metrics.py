# Script to load the pickle data and plot it

from speck_rem import *

target_folder = "D:/Research/SpeckleRemoval/Data/2018_11_22/three/planar_fixed_freq_manual/comp/"
data_filename = "data.pkl"

plt.rcParams.update({'font.size': 14})

with open(os.path.join(target_folder, data_filename), "rb") as f:
    speckle_data = pickle.load(f)

number_images = len(speckle_data["sc_avg"])

#Contrast
t = np.arange(1, number_images+1)
plt.plot(t, 1.0/np.sqrt(t), 'k--')
plt.plot(t, speckle_data["sc_avg"]/max(speckle_data["sc_avg"]), 'bo')
plt.plot(t, speckle_data["sc_std"]/max(speckle_data["sc_std"]), 'r*')
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
