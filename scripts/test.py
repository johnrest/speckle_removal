# Testing script for testing the tests
from speck_rem import *

pm = RandomPhaseMask()
pm.optimize()

display_image(pm.image_array, 0.5, "Mask")