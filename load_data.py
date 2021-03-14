from os.path import join, exists
from os import listdir, makedirs
import numpy as np
import nibabel as nib
from PIL import Image


def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if not arr_max:
        arr[True] = 0
        return arr
    arr -= arr_min
    arr *= (2**16-1) / (arr_max-arr_min)
    return arr

def save_slices(np_volume, save_path, file_name):
    if not exists(save_path):
        makedirs(save_path)
    n = np_volume.shape[2]
    for i in range(n):
        slab = np_volume[:, :, i]
        img = Image.fromarray(slab)
        new_file_name = file_name + '_slice_{}.png'.format(i)
        img.save(join(save_path, new_file_name))


def save_slices_8bit(np_volume, save_path, file_name):
    if not exists(save_path):
        makedirs(save_path)
    n = np_volume.shape[2]
    for i in range(n):
        slab = np_volume[:, :, i]
        norm_slab = normalize(slab)
        img = Image.fromarray(norm_slab.astype(np.uint16))
        new_file_name = file_name + '_slice_{}.png'.format(i)
        img.save(join(save_path, new_file_name))


def read_training_data(data_path, save_path):
    '''Read training data'''
    for f in listdir(join(data_path, 'imagesTr')):
        if '.DS_Store' in f:
            continue
        img_array = nib.load(join(data_path, 'imagesTr', f)).get_fdata()
        lbl_array = nib.load(join(data_path, 'labelsTr', f)).get_fdata()
        file_name = f.split('/')[-1].split('.')[0]
        save_slices_8bit(img_array, join(save_path, 'training_images'), file_name)
        save_slices_8bit(lbl_array, join(save_path, 'training_labels'), file_name)


def read_testing_data(data_path, save_path):
    '''Read testing data'''
    imgs = []
    for f in listdir(join(data_path, 'imagesTs')):
        if '.DS_Store' in f:
            continue
        img_array = nib.load(join(data_path, 'imagesTs', f)).get_fdata()
        file_name = f.split('/')[-1].split('.')[0]
        save_slices_8bit(img_array, join(save_path, 'test_images'), file_name)


def read_for_classification(data_path, save_path):
    for f in listdir(join(data_path, 'imagesTr')):
        if '.DS_Store' in f:
            continue
        img_array = nib.load(join(data_path, 'imagesTr', f)).get_fdata()
        lbl_array = nib.load(join(data_path, 'labelsTr', f)).get_fdata()
        file_name = f.split('/')[-1].split('.')[0]
        n = lbl_array.shape[2]
        for i in range(n):
            img_slab = img_array[:, :, i]
            norm_img_slab = normalize(img_slab)
            img = Image.fromarray(norm_img_slab.astype(np.uint16))
            new_file_name = file_name + '_slice_{}.png'.format(i)
            label_slab = lbl_array[:, :, i]
            if np.sum(label_slab) == 0:
                img.save(join(save_path, '0', new_file_name))
            else:
                img.save(join(save_path, '1', new_file_name))
                



data_path = "C:/Users/Karol/Desktop/ML4HC/Project_1/ml4h_proj1_colon_cancer_ct"
save_path = "C:/Users/Karol/Desktop/ML4HC/Project_1/images/classification"
read_training_data(data_path, save_path)
read_testing_data(data_path, save_path)
read_for_classification(data_path, save_path)

# print(type(test_imgs[0]))