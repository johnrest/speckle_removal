# Script to select profiles from a list of images

from speck_rem import *


def onMouse(event, x, y, flags, param):
    global posList
    if event == cv2.EVENT_LBUTTONDOWN:
        global posList
        print(x, y)
        posList.append((x, y))


target_folder = r"D:\Research\SpeckleRemoval\Data\2018_11_22\three\planar_fixed_freq_manual/temp\rec2/"
name_mask = "rec_*"

images_list = get_list_images(target_folder, name_mask)

# Plot first image for reference
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

points = np.array(posList)                       # convert to numpy for other usages
print("Stored positions are: ", points)
points = points.T

profile_data = image_profile(images_list[0:2], points)

fig = plt.figure()
plt.plot(profile_data[0,:], 'k--')
plt.plot(profile_data[1,:], 'r--')

plt.show()