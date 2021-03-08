import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence


def weighted_bce(y_true, y_pred, weight1=1, weight0=1 ) :
    weights =  (1.0 - y_true) * weight0 +  y_true * weight1 
    bce = K.binary_crossentropy(y_true, y_pred)
    w_bce = K.mean(weights * bce)

    return w_bce


def IoU(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred))
  union = K.sum(y_true)+K.sum(y_pred) - intersection
  iou = K.mean((intersection + smooth) / (union + smooth))
  return iou


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def normalize(arr, th=None):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if not arr_max:
        arr = 0
        return arr
    arr = arr / (arr_max-arr_min)
    if th:
        arr[arr<=th] = 0
        arr[arr>th] = 1
    return arr


class MyGenerator(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, labels_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.labels_img_paths = labels_img_paths

    def __len__(self):
        return len(self.labels_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_label_img_paths = self.labels_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = Image.open(path)
            arr = np.array(img.resize(self.img_size, resample=Image.NEAREST))
            x[j] = normalize(arr)
        y = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_label_img_paths):
            img = Image.open(path)
            arr = np.array(img.resize(self.img_size, resample=Image.NEAREST))
            y[j] = normalize(arr, th=0.5)
        return x, y
