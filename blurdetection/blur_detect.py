# import packages
import matplotlib.pyplot as plt
import numpy as np


def fft_blur_detect(image, title="", size=60, thresh=1, filter_type="s", vis=False):
    """

    :rtype: object
    """
    # calculate image center using dimensions
    h, w = image.shape
    cX, cY = int(w / 2.0), int(h / 2.0)

    # compute FFT and shift DC component to center
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # zero the DC component (low frequencies)
    if filter_type == "s":
        fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    elif filter_type == "c":
        for y in range(len(fft_shift)):
            for x in range(len(fft_shift[0])):
                if distance((x, y), (cX, cY)) < size:
                    fft_shift[y, x] = 0
    fft_unshift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_unshift)

    # compute magnitude spectrum of reconstructed image
    magnitude = np.log(np.abs(recon))
    # compute mean of magnitudes
    mean = np.mean(magnitude)

    # to visualize filtered magnitude spectrum
    if vis:
        # compute the magnitude spectrum of the transform
        mag = np.log(np.abs(fft_shift), where=np.abs(fft_shift) > 0)
        # display the original input image
        (fig, ax) = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Original")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(mag, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # display the magnitude image
        ax[2].imshow(np.abs(recon), cmap="gray")
        ax[2].set_title("After filtering, Mean = " + str(np.around(mean, 4)))
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        # show our plots
        plt.savefig('filtered/f' + filter_type + "_" + title, bbox_inches='tight')

    # effect of size
    if not vis:
        mag = np.log(np.abs(fft_shift), where=np.abs(fft_shift) > 0)

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, figsize=(10, 4))
        # display the magnitude image
        ax[0].imshow(mag, cmap="gray")
        ax[0].set_title("Magnitude Spectrum")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(np.abs(recon), cmap="gray")
        ax[1].set_title("After filtering")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.title("Circle radius = " + str(size) + " pixels, Mean = " + str(np.around(mean, 4)))
        plt.savefig('size/' + filter_type + "_" + title[:-4] + str(size) + ".jpg", bbox_inches='tight')

    # blurry if mean < threshold
    return mean, mean <= thresh


def distance(point, center):
    return np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
