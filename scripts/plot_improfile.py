# Script to select profiles from a list of images

from speck_rem import *
plt.rcParams.update({'font.size': 16})


def onMouse(event, x, y, flags, param):
    global posList
    if event == cv2.EVENT_LBUTTONDOWN:
        global posList
        print(x, y)
        posList.append((x, y))


target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual\improfiles/"
name_mask = "average_*"

images_list = get_list_images(target_folder, name_mask)
print(images_list, sep="\n")
# Plot first image for clicks
image = Image()
image.read_image_file_into_array(images_list[0])

# Select two points with click over image
posList = []

# display_image(image.image_array, 1, "Profile")
cv2.namedWindow("Profile")
cv2.setMouseCallback('Profile', onMouse)

while True:
    cv2.imshow("Profile", image.image_array / np.max(image.image_array))
    key = cv2.waitKey(1) & 0xFF

    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break

points = np.array(posList)  # convert to numpy for other usages
print("Stored positions are: ", points)
points = points.T

profile_data = image_profile(images_list, points)

fig = plt.figure(figsize=(8,6))
plt.plot(profile_data[0, :], 'g--', label="64")
plt.plot(profile_data[1, :], 'b--', label="128")
plt.plot(profile_data[2, :], 'r--', label="512")
plt.legend()
plt.xlabel("$Distance\ along\ profile\ [pix]$")
plt.ylabel("$I$")

format = '.svg'

plt.savefig(os.path.join(target_folder, "improfile" + format), bbox_inches="tight")

plt.show()