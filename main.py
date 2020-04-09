import numpy as np
from image_loader import ImageLoader as IL
import matplotlib.pyplot as plt

# ===================================== LOAD IMAGE =====================================
origin_image = IL("lenna.png")()

# change to gray scale.
origin_image_gray = 0.2989 * origin_image[:, :, 0] + 0.5870 * origin_image[:, :, 1] + 0.1140 * origin_image[:, :, 2]

# ===================================== ADD GAUSSIAN NOISE =====================================
# stdev = 44.02  # SNR_db ~ 10 (SNR : 10)
# stdev = 13.3  # SNR_db ~ 20 (SNR : 100)
# stdev = 4.21  # SNR_db ~ 30 (SNR : 1000)

SNRdb_10_20_30_stdev = {"10": 44.02, "20": 13.3, "30": 4.21}

for SNR_db in SNRdb_10_20_30_stdev:
    stdev = SNRdb_10_20_30_stdev[SNR_db]

    G_1 = np.max(origin_image_gray)
    print("G-1 = {}".format(G_1))

    # Box-Muller transform
    f_z1 = lambda x, y, z: (z * np.cos(2 * np.pi * y)) * np.sqrt(-2 * np.log(x))
    f_z2 = lambda x, y, z: (z * np.sin(2 * np.pi * y)) * np.sqrt(-2 * np.log(x))


    def additive_pixel_cal(origin_pixel_val, z_val, param_g_1):
        f_prime = origin_pixel_val + z_val

        # print("f_prime : ", f_prime, ", origin_pixel_val : ", origin_pixel_val)

        if f_prime < 0:
            return 0
        elif f_prime > param_g_1:
            return param_g_1
        else:
            return f_prime


    noise_added = np.zeros(shape=origin_image_gray.shape)

    print("processing...")

    for _x in range(origin_image_gray.shape[0]):
        for _y in range(origin_image_gray.shape[1]):

            r, pi = np.random.rand(2)
            z1 = f_z1(r, pi, stdev)
            z2 = f_z2(r, pi, stdev)

            noise_added[_x, _y] = additive_pixel_cal(origin_image_gray[_x, _y], z1, G_1)
            if _y + 1 < origin_image_gray.shape[1]:
                noise_added[_x, _y + 1] = additive_pixel_cal(origin_image_gray[_x, _y + 1], z2, G_1)

    E = np.mean(np.square(noise_added - origin_image_gray))
    F = np.mean(np.square(origin_image_gray))

    SNR = F / E

    print("SNR :", SNR)
    print("stdev :", stdev)

    plt.rcParams["figure.figsize"] = (8, 4.2)
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(origin_image_gray, 'gray')
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Noise(SNRdb:{0:0.1f}, stdev:{1:4.2f})".format(10 * np.log10(SNR), stdev))
    plt.imshow(noise_added, 'gray')
    plt.axis("off")
    plt.savefig("./SNRdb{}.png".format(SNR_db))
    plt.show()

    # ===================================== DRAW HISTOGRAM =====================================
    h_f = np.zeros(256)
    for _x in range(noise_added.shape[0]):
        for _y in range(noise_added.shape[1]):
            h_f[int(noise_added[_x, _y])] += 1
    plt.title("SNRdb {0:0.1f} brightness histogram".format(10 * np.log10(SNR)))
    plt.bar(list(range(len(h_f))), h_f)
    plt.xlim([0, 255])
    plt.savefig("./SNRdb{}_histogram.png".format(SNR_db))
    plt.show()
