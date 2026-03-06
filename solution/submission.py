import cv2
import numpy as np


class Model:
    def fit(self, train_images, train_masks, train_labels=None):
        return self

    def predict(self, test_images, sample_ids=None):
        preds = []
        for image in test_images:
            img = np.array(image.convert("L"))
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            _, mask = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            preds.append(mask)
        return preds


def get_model():
    return Model()
