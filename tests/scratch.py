import numpy as np
import numpy.ma as ma
import imageio
import os

a = np.arange(16).reshape(2, 2, 4).astype('uint8')

for i in np.arange(a.shape[2]):
    imageio.imwrite('test_image_' + str(i) + '.png', a[:, :, i].astype('uint8'))

assert os.path.exists('test_image_3.png')
assert not os.path.exists('test_image_4')
assert imageio.imread('test_image_2.png')[1, 1] == 14
