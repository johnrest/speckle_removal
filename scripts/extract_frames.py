# Script to extract frames from an avi video produced by the triggered acquisition

from speck_rem import *

target_folder = "C:/Users/itm/Desktop/DH/2018_10_31/test_random_patternC"
video_filename = "holo.avi"
image_filename_mask = "holo_"

print("Extracting frames...")
extract_frames_from_video(target_folder, video_filename, image_filename_mask)
print("Finished.")

