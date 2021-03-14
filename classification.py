import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, InputLayer, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import json
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from utils import normalize


data_path = "C:/Users/Karol/Desktop/ML4HC/Project_1/images/classification"

target_size = (256, 256)
epochs_num = 150
batch = 16
l1 = 0.00001
l2 = 0.000001
dropout = 0.5


train_generator = ImageDataGenerator(preprocessing_function=normalize,
                                     width_shift_range=0.05,
                                     height_shift_range=0.05,
                                     zoom_range=0.001,
                                     horizontal_flip=True,
                                     rotation_range=30,
                                     validation_split=0.2 )

train_data = train_generator.flow_from_directory(directory=data_path,
                                                    batch_size=batch,
                                                    target_size=target_size,
                                                    color_mode='grayscale',
                                                    shuffle=True,
                                                    class_mode='binary')


# verify generator
# images, labels = train_data.__getitem__(0)
# for x, y in zip(images, labels):
#     print(y)
#     plt.imshow(x[:,:,0], cmap='gray')
#     plt.show()


reg = l1_l2(l1=l1, l2=l2)

model = Sequential()
model.add(InputLayer([target_size[0], target_size[1], 1]))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(128, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
adam = Adam(lr=0.001)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

drop_alpha = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
checkpoint = ModelCheckpoint('colon_classification.h5', monitor='val_acc', verbose=0, 
							  save_best_only=True, save_weights_only=False, mode='auto', period=1)

#Train
history = model.fit(train_data,
                    epochs=epochs_num,
                    callbacks=[drop_alpha, checkpoint],
                    class_weight={0: 1, 1: 10})

# saving the model
model.save('tumor_classification.h5')
print("Saved model to disk")

print(history.history)
with open('history.json', 'w') as f:
    json.dump(history.history, f)

# Visualizing results of the training
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_num)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
