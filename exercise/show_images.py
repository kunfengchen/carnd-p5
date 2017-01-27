import matplotlib.pyplot as plt
import glob
import scipy.misc

img1 = scipy.misc.imread('../data/vehicles/GTI_Far/image0758.png')
img2 = scipy.misc.imread('../data/non-vehicles/GTI/image3807.png')

plt.subplot(121)
plt.title('Car')
plt.imshow(img1)
plt.subplot(122)
plt.title('Not Car')
plt.imshow(img2)
plt.show()
