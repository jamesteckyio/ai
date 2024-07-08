import os

import tensorflow as tf
from tensorflow import keras
import pathlib                               
import matplotlib.pylab as plt
import cv2
import numpy as np
import threading
from transformers import ViTForImageClassification


def display_image(resize_img):
    plt.figure()
    plt.imshow(resize_img)
    plt.colorbar()
    plt.grid(True)
    plt.xlabel(class_names[0])
    plt.show()
    
print(tf.version.VERSION)

probability_model = tf.keras.models.load_model('my_model.h5')

class_names = ['cats', 'dogs']
test_dir=pathlib.Path("./dog_cat_dataset/testing_set")
cat_test= list(test_dir.glob('cats/*'))
dog_test= list(test_dir.glob('dogs/*'))

predict_index = 45
# Grab an image from the test dataset.
image = cat_test[predict_index]
img = cv2.imread(str(image))

resize_img = cv2.resize(img, (224, 224))
resize_img = resize_img / 255.0

threading.Thread(target=display_image(resize_img), daemon=True).start()


# Add the image to a batch where it's the only member.
resize_img = (np.expand_dims(resize_img,0))

print(resize_img.shape)

predictions_single = probability_model.predict(resize_img)
predictions_label = class_names[np.argmax(predictions_single[0])]

print(predictions_single)
print(predictions_label)

