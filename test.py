from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

# Open the image using GDAL
im = gdal.Open('ensta_2015.jpg')

# Get image dimensions
nx = im.RasterXSize
ny = im.RasterYSize
nb = im.RasterCount

# Initialize the image array
image = np.zeros((ny, nx, nb), dtype=np.float32)

# Read each band into the image array and normalize to [0, 1]
image[:, :, 0] = im.GetRasterBand(1).ReadAsArray() / 255.0
image[:, :, 1] = im.GetRasterBand(2).ReadAsArray() / 255.0
image[:, :, 2] = im.GetRasterBand(3).ReadAsArray() / 255.0

# Plotting the image
plt.figure()
plt.xlim([500, 1000])
plt.ylim([1200, 800])
plt.imshow(image)
plt.show()
