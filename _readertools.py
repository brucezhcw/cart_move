import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
import win_unicode_console
win_unicode_console.enable()

def batch_generator(picture_path, label_path, batch_size):
    label_resource = os.listdir(path=label_path)
    picture_resource = os.listdir(path=picture_path)
    num_data = range(len(label_resource))
    num_data = np.array(num_data)
    if batch_size > len(label_resource):
        raise ValueError("please input a batch_size smaller than: %f", len(label_resource))
    while True:
        np.random.shuffle(num_data)
        for k in range(len(picture_resource) // batch_size):
            labels = []
            pictures = []
            for i in range(batch_size):
                picture = Image.open(os.path.join(picture_path, picture_resource[num_data[k * batch_size + i]]))
                picture = picture.convert('L')
                picture = np.array(picture)
                pictures.append(picture)
                label = np.loadtxt(os.path.join(label_path, label_resource[num_data[k * batch_size + i]]))
                label = np.array(label)
                #label_ = np.zeros(26, int)
                #for j in range(len(label)):
                #    label_[j] = label[j]
                labels.append(label)
            labels = np.array(labels)
            #pictures = np.array(pictures)
            yield pictures, labels
