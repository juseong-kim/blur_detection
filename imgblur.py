# import packages
from blurdetection.blur_detect import fft_blur_detect
import numpy as np
import argparse
import imutils
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input image")
ap.add_argument("-t", "--thresh", type=int, default=1,
                help="blur threshold")
ap.add_argument("-v", "--vis", type=int, default=-1,
                help="visualize intermediate steps")
ap.add_argument("-d", "--test", type=int, default=-1,
                help="progressive blur")
ap.add_argument("-s", "--size", type=int, default=60,
                help="highpass filter radius")
ap.add_argument("-f", "--filter", type=str, default="s",
                help="square (s) or circle (c) filter")
args = vars(ap.parse_args())

# load input image and resize
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=500)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# apply blur detector
(mean, blurry) = fft_blur_detect(orig, title=args["image"][7:], size=args["size"], thresh=args["thresh"],
                                 filter_type=args["filter"], vis=args["vis"] > 0)

# annotate blurry or not
image = np.dstack([orig] * 3)
color = (40, 43, 248) if blurry else (25, 194, 45)
text = "Blur detected ({:.4f})" if blurry else "No blur detected ({:.4f})"
text = text.format(mean)
cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
print("[INFO] {}".format(text))

# save output as jpg file to output folder
filename = 'output/o' + args["filter"] + "_" + str(np.around(mean, 4)) + "_" + args["image"][7:]
cv2.imwrite(filename, image)

# test blur detection for varying degrees of Gaussian blur
if args["test"] > 0:
    # loop over various blur radii
    for radius in range(1, 16, 2):
        # clone the original grayscale image
        image = orig.copy()
        # if the kernel radius is greater than zero
        if radius > 0:
            # blur the input image with Gaussian kernel of specified radius
            image = cv2.GaussianBlur(image, (radius, radius), 0)
            # apply blur detection algorithm
            (mean, blurry) = fft_blur_detect(image, title=args["image"][7:], size=args["size"],
                                             thresh=args["thresh"], filter_type=args["filter"], vis=args["vis"] > 0)
            # annotate result of algorithm on image
            image = np.dstack([image] * 3)
            color = (40, 43, 248) if blurry else (25, 194, 45)
            text = "Blur detected ({:.4f})" if blurry else "No blur detected ({:.4f})"
            text = text.format(mean)
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            print("[INFO] Kernel: {}, Result: {}".format(radius, text))

        # save the image as jpg file to gaussian folder
        filepath = "gaussian/" + args["image"][7:-4] + str(radius) + ".jpg"
        cv2.imwrite(filepath, image)
