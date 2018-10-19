# Testing script for testing the tests
from speck_rem import *

sc = int(1280/320)
batch = compute_pattern_batch(scale=sc, batch_length=(sc*sc)/2)

total = np.zeros(batch[0].shape)
for item in batch:
    pass

cv2.waitKey(0)
cv2.destroyAllWindows()

