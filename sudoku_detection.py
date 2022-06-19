import cv2
import numpy as np

def preprocess(image, debug = False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = apply_brightness_contrast(image, 0, 100)
    if debug:
        cv2.imshow("Gray", image)
    image = cv2.Canny(image, 60, 200)
    image = cv2.dilate(image, (5, 5), iterations=5)
    if debug:
        cv2.imshow("Canny", image)
    return image

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def get_contour(img_canny):
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        accuracy = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, accuracy, True)
        if len(approx) == 4:
            if area > 10000 and area > max_area:
                max_area = area
                max_contour = approx
    return max_contour

def reorder(points: np.ndarray):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(axis=1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def pipeline(image, debug = False):
    # image = cv2.resize(image, (1000, 592))
    preprocessed = preprocess(image, debug)
    biggest = get_contour(preprocessed)
    if biggest is None:
        return None
    if debug:
        im = image.copy()
        cv2.drawContours(im,[biggest], -1, (0,0,255), 3)
        cv2.imshow("Biggest", im)
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    return pts1

