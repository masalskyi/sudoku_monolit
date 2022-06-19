import cv2
import numpy as np


VERT_SHIFT = 36
HOR_SHIFT =20


def draw_on_image(image, data, size):
    size *= size
    data = np.array(data).reshape(size, size)
    for i in range(size):
        for j in range(size):
            if data[i, j] != 0:
                cv2.putText(image,
                            str(data[i, j]),
                            (int(j * image.shape[0] / size + HOR_SHIFT), int(i * image.shape[0] / size + VERT_SHIFT)),
                            cv2.FONT_ITALIC,
                            fontScale=1,
                            color=(255,0,0),
                            thickness=3)
    return image