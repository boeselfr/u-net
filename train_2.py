from glob import glob
from os.path import join
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *
from matplotlib import pyplot as plt
from model import unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import segmentation_models as sm
from utils import normalize, normalize_classif

def validate_generator(gen):
    images, labels = next(gen)
    for x, y in zip(images, labels):
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(x, cmap='gray')
        ax[1].imshow(y, cmap='gray')
        plt.show()

def unet_train():

    BACKBONE = 'efficientnetb0'
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1)
    image_datagen = ImageDataGenerator(**aug_dict, preprocessing_function=normalize_classif)
    mask_datagen = ImageDataGenerator(**aug_dict, preprocessing_function=normalize_classif)

    val_image_datagen = ImageDataGenerator({}, preprocessing_function=normalize_classif)
    val_mask_datagen = ImageDataGenerator({}, preprocessing_function=normalize_classif)
    seed = 1
    STEP_SIZE_TRAIN = TRAIN_SIZE // BATCH_SIZE
    STEP_SIZE_VALID = VALID_SIZE // BATCH_SIZE

    image_generator = image_datagen.flow_from_directory(join(data_path, 'classification'), target_size=IMG_SIZE, classes=['training_images'],
                                                        class_mode=None, seed=seed, batch_size=BATCH_SIZE,
                                                        color_mode='rgb')
    mask_generator = mask_datagen.flow_from_directory(join(data_path, 'classification'), classes=['training_labels'], target_size=IMG_SIZE,
                                                      class_mode=None, seed=seed, batch_size=BATCH_SIZE,
                                                      color_mode='rgb')

    val_image_generator = val_image_datagen.flow_from_directory(data_path, classes=['validation_images_cancer'], target_size=IMG_SIZE,
                                                        class_mode=None, seed=seed, batch_size=BATCH_SIZE,
                                                        color_mode='rgb')
    val_mask_generator = val_mask_datagen.flow_from_directory(data_path, classes=['validation_labels_cancer'], target_size=IMG_SIZE,
                                                      class_mode=None, seed=seed, batch_size=BATCH_SIZE,
                                                      color_mode='rgb')

    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)
    model_checkpoint = ModelCheckpoint('unet_16_efc_all.hdf5', monitor='loss', save_best_only=True, period=1)
    tensorboard_callback = TensorBoard(log_dir='./logs/unet_16')
    drop_alpha = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    model.compile(optimizer=Adam(lr=learning_rate), loss=wbce_loss, metrics=my_metrics)
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, verbose=1,
              steps_per_epoch=STEP_SIZE_TRAIN,validation_steps=STEP_SIZE_VALID,
                        callbacks=[model_checkpoint, tensorboard_callback, drop_alpha])

def train(validate_input=False):
    """aug_dict={}
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict, preprocessing_function=normalize)
    image_generator = image_datagen.flow_from_directory(
        DATA_FOLDER,
        classes=['training_images_cancer'],
        class_mode=None,
        color_mode='grayscale',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE)
    print(image_generator.n)
    return"""
    training_generator = MyAugGenerator(BATCH_SIZE, data_path, 'cancer_images', 'cancer_masks',
                                        aug_dict,
                                        IMG_SIZE)
    val_generator = MyAugGenerator(BATCH_SIZE, data_path, 'validation_images_cancer', 'validation_labels_cancer', {},
                                   IMG_SIZE)
    STEP_SIZE_TRAIN = TRAIN_SIZE // BATCH_SIZE
    STEP_SIZE_VALID = VALID_SIZE // BATCH_SIZE

    if validate_input:
        validate_generator(training_generator)
    model = unet(IMG_SIZE, dropout)
    model.compile(optimizer=Adam(lr=learning_rate), loss=wbce_loss, metrics=my_metrics)
    model_checkpoint = ModelCheckpoint('unet_16.hdf5', monitor='loss', save_best_only=True, period=1)
    tensorboard_callback = TensorBoard(log_dir='./logs/unet_16')
    drop_alpha = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    model.fit(training_generator,
              validation_data=val_generator,
              steps_per_epoch=STEP_SIZE_TRAIN,
              epochs=EPOCHS,
              validation_steps=STEP_SIZE_VALID,
              callbacks=[model_checkpoint, tensorboard_callback, drop_alpha])


def predict(show=True):
    model = load_model('unet_16_efc.hdf5', compile=False)
    model.compile(optimizer=Adam(lr=learning_rate), loss=wbce_loss, metrics=my_metrics)

    for path in test_paths:
        print(path)
        img = Image.open(path)
        input_arr = np.array(img.resize(IMG_SIZE, resample=Image.NEAREST))
        input_arr = input_arr.reshape([input_arr.shape[0], input_arr.shape[1], 1])
        input_arr = normalize_classif(input_arr)
        input_arr = input_arr.reshape([1, input_arr.shape[0], input_arr.shape[1], input_arr.shape[2]])
        prediction = model.predict(input_arr, batch_size=1)[0, :, :, 0]
        #print(prediction)
        p = np.asarray(prediction)
        ct_25 = (p > 0.25).sum()
        ct_5 = (p > 0.5).sum()
        ct_75 = (p > 0.75).sum()

        print(f'> 0.25 {ct_25}')
        print(f'> 0.5 {ct_5}')
        print(f'> 0.75 {ct_75}')

        if show:
            plt.imshow(prediction)
            plt.show()

        img = Image.fromarray(prediction*255.0)
        file_name = path.split('/')[-1]
        img.convert("L").save(join(result_path, file_name))


############################################################################################

# Macroparameters

data_path = '/cluster/scratch/fboesel/images'
result_path = '../unet_result'
BATCH_SIZE = 64
EPOCHS = 250
IMG_SIZE = (224, 224)
learning_rate = 1e-3
dropout = 9 * [0.25]
weight_zeros = 0.11
weight_ones = 5
aug_dict = {
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'zoom_range': 0.001,
    'horizontal_flip': True,
    'rotation_range': 30
}
TRAIN_SIZE = len(glob(join(data_path, 'training_images', '*.png')))
VALID_SIZE = len(glob(join(data_path, 'validation_labels_cancer', '*.png')))

test_paths = glob(join(data_path, 'classification', 'test_images', '*.png'))
############################################################################################

wbce_loss = lambda y_true, y_pred: weighted_bce(y_true, y_pred, weight0=weight_zeros, weight1=weight_ones)
my_metrics = [f1, IoU]

sm.set_framework('tf.keras')

#unet_train()
#train(True)
predict()