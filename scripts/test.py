# Testing script for testing the tests
from speck_rem import *
from matplotlib import cm as CM                 #Move to ini file

plt.rcParams.update({'font.size': 14})

target_folder = "C:/Users/itm/Desktop/DH/2018_11_08/three"

data_planar = np.load(os.path.join(target_folder, "planar_data.npz"))
data_walsh = np.load(os.path.join(target_folder, "walsh_data.npz"))
data_rand = np.load(os.path.join(target_folder, "random_data.npz"))

sc_planar = data_planar[data_planar.files[1]]
sc_walsh = data_walsh[data_walsh.files[1]]
sc_rand = data_rand[data_rand.files[1]]


#Contrast
t = np.arange(1, len(sc_planar)+1)
plt.plot(t, 1.0/np.sqrt(t), 'k--')
plt.plot(t, sc_planar, 'bs')
plt.plot(t, sc_walsh, 'go')
plt.plot(t, sc_rand, 'rX')
plt.xlabel('N')
plt.xticks(t)

plt.legend(('Theoretical', 'Planes', 'Walsh', "Random"),
           bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0)

plt.savefig(os.path.join(target_folder, "coeff.png"), bbox_inches="tight")
plt.show()



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