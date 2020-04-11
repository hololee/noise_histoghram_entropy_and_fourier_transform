import numpy as np
from image_loader import ImageLoader as IL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================================== LOAD IMAGE =====================================
origin_image = IL("lenna.png")(resize_factor=(128, 128))

# change to gray scale.
origin_image_gray = 0.2989 * origin_image[:, :, 0] + 0.5870 * origin_image[:, :, 1] + 0.1140 * origin_image[:, :, 2]
origin_image_gray = origin_image_gray.astype(np.int)


# ===================================== DEFINE FUNCTIONS =====================================
def discrete_fourier_transform(f_array, index):
    _F = [
        np.mean([f * np.exp((-2 * np.pi * 1j * index_f * index) / len(f_array)) for index_f, f in enumerate(f_array)])]
    return _F


def fft_cooley_tukey_algorithm():
    pass


transformed = np.zeros(shape=origin_image_gray.shape, dtype=np.float)

for i in range(transformed.shape[0]):
    for j in range(transformed.shape[1]):
        print("processing... ({},{})".format(i, j))

        first_summation = [discrete_fourier_transform(origin_image_gray[index1, :], i) for index1 in
                           range(transformed.shape[0])]

        # power spectrum log.
        transformed[i, j] = np.log(
            np.abs(np.squeeze(discrete_fourier_transform(np.squeeze(np.array(first_summation)), j))) ** 2)
        print('val: ', transformed[i, j])

# ===================================== DRAW TARGET IMAGE =====================================
plt.title("Target image")
plt.axis("off")
plt.imshow(origin_image_gray, 'gray')
plt.savefig("./results/Target_image.png")
plt.show()

# ===================================== DRAW 2D FIGURE =====================================
plt.title("Power spectrum 2D")
plt.axis("off")
plt.imshow(transformed)
plt.colorbar()
plt.savefig("./results/Power_spectrum_2D.png")
plt.show()

# ===================================== DRAW 3D FIGURE =====================================
fig = plt.figure()
ax = Axes3D(fig)

axis_x = list(range(transformed.shape[0]))
axis_y = list(range(transformed.shape[1]))
coordinate = np.meshgrid(axis_x, axis_y)

# ax.scatter(coordinate[0], coordinate[1], transformed, cmap='viridis', edgecolor='none')
# ax.contour3D(coordinate[0], coordinate[1], transformed, 20, cmap=plt.cm.rainbow, edgecolor='none')
ax.plot_surface(coordinate[0], coordinate[1], transformed, cmap='viridis', alpha=1, edgecolor='none')
plt.title("Power spectrum 3D")
plt.savefig("./results/Power_spectrum_3D.png")
plt.show()

# ===================================== CHANE DATA TO CENTERED =====================================
centered_transformed = np.zeros(transformed.shape)
half_h = (transformed.shape[0] // 2)
half_w = (transformed.shape[1] // 2)
# LT
centered_transformed[0:half_h, 0:half_w] = transformed[half_h:, half_w:]

# RB
centered_transformed[half_h:, half_w:] = transformed[0:half_h, 0:half_w]

# RT
centered_transformed[0:half_h, half_w:] = transformed[half_h:, 0:half_w]

# LB
centered_transformed[half_h:, 0:half_w] = transformed[0:half_h, half_w:]

# ===================================== DRAW CENTERED 2D FIGURE =====================================
plt.title("Power spectrum centered 2D")
plt.axis("off")
plt.imshow(centered_transformed)
plt.colorbar()
plt.savefig("./results/Power_centered_spectrum_2D.png")
plt.show()

# ===================================== DRAW CENTERED 3D FIGURE =====================================
fig = plt.figure()
ax = Axes3D(fig)

axis_x = list(range(centered_transformed.shape[0]))
axis_y = list(range(centered_transformed.shape[1]))
coordinate = np.meshgrid(axis_x, axis_y)

# ax.scatter(coordinate[0], coordinate[1], transformed, cmap='viridis', edgecolor='none')
# ax.contour3D(coordinate[0], coordinate[1], transformed, 20, cmap=plt.cm.rainbow, edgecolor='none')
ax.plot_surface(coordinate[0], coordinate[1], centered_transformed, cmap='viridis', alpha=1, edgecolor='none')
plt.title("Power spectrum centered 3D")
plt.savefig("./results/Power_centered_spectrum_3D.png")
plt.show()
