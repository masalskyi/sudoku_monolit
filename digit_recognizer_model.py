import numpy as np
import tensorflow as tf
import cv2

from constants import MODEL_PATH, THRESHOLD, IMG_SIZE, BIGGER_IMG_SIZE

def image_preprocess(image, size):
    image = cv2.resize(image, (IMG_SIZE*size*size, IMG_SIZE*size*size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype("float32") / 255
    boxes = []
    for r in np.vsplit(image, size * size):
        for box in np.hsplit(r, size * size):
            boxes.append(box)
    return np.array(boxes).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

class DigitRecognizerModel:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.threshold = THRESHOLD

    def predict(self, image, size):
        boxes = image_preprocess(image, size)
        predictions = self.model.predict(boxes, verbose=0, batch_size=size**4)
        classes = [0 for _ in range(len(predictions))]
        for i in range(len(predictions)):
            classes[i] = self._get_class(predictions[i])
        return classes

    def _get_class(self, prediction):
        arg = np.argmax(prediction)
        ma = np.amax(prediction)
        if ma < self.threshold:
            return 0
        else:
            return arg.item()