from glob import glob
from os.path import join
from utils import *
from matplotlib import pyplot as plt
from model import unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np


def validate_generator(gen):
    images, labels = next(gen)
    for x, y in zip(images, labels):
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(x, cmap='gray')
        ax[1].imshow(y, cmap='gray')
        plt.show()


def train(validate_input=False):
    training_generator = MyAugGenerator(BATCH_SIZE, train_path, 'training_images', 'training_labels', aug_dict, IMG_SIZE)
    val_generator = MyAugGenerator(BATCH_SIZE, train_path, 'training_images', 'training_labels', {}, IMG_SIZE)
    if validate_input:
        validate_generator(training_generator)
    model = unet(IMG_SIZE, dropout)
    model.compile(optimizer=Adam(lr=learning_rate), loss=wbce_loss, metrics=my_metrics)
    model_checkpoint = ModelCheckpoint('unet_checkpoint.hdf5', monitor='loss', save_best_only=True)
    model.fit(training_generator, 
              validation_data=val_generator, 
              epochs=EPOCHS, 
              callbacks=[model_checkpoint])


def predict(show=True):
    model = load_model('unet_checkpoint.hdf5',  compile=False)
    model.compile(optimizer=Adam(lr=learning_rate), loss=wbce_loss, metrics=my_metrics)

    for path in test_paths:
        print(path)
        img = Image.open(path)
        input_arr = np.array(img.resize(IMG_SIZE, resample=Image.NEAREST))
        input_arr = input_arr.reshape([input_arr.shape[0], input_arr.shape[1], 1])
        input_arr = normalize(input_arr)
        input_arr = input_arr.reshape([1, input_arr.shape[0], input_arr.shape[1]])
        prediction = model.predict(input_arr, batch_size=1)[0, :, :, 0]

        if show:
            plt.imshow(prediction)
            plt.show()

############################################################################################

# Macroparameters
BATCH_SIZE = 32
EPOCHS = 1
IMG_SIZE = (256, 256)
learning_rate = 1e-3
dropout = 9 * [0.25]
weight_zeros = 1
weight_ones = 1
aug_dict={
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'zoom_range': 0.001,
        'horizontal_flip': True,
        'rotation_range': 30
}


# Paths
train_path = "C:/Users/Karol/Desktop/ML4HC/Project_1/images"

test_paths = glob(join(train_path, 'test_images', '*.png'))

result_path = "C:/Users/Karol/Desktop/unet_result"

############################################################################################

wbce_loss = lambda y_true, y_pred: weighted_bce(y_true, y_pred, weight0=weight_zeros, weight1=weight_ones)
my_metrics = [f1, IoU]

# train(validate_input=True)
predict()
