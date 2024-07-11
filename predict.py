import tensorflow as tf
import pathlib                               
import numpy as np
import matplotlib.pylab as plt
import cv2
import keras
from patches import Patches, PatchEncoder
image_size = 200
class_names = ['cats', 'dogs']
# Load the  model
model = keras.models.load_model('./models/vit.keras')

test_dir=pathlib.Path("./dog_cat_dataset/testing_set")
cat_test= list(test_dir.glob('cats/*'))
dog_test= list(test_dir.glob('dogs/*'))

print("Cat length: ", len(cat_test))
print("Dog length: ", len(dog_test))


predict_index = 79
# Grab an image from the test dataset.
image = cat_test[predict_index]
img = cv2.imread(str(image))

resize_img = cv2.resize(img, (image_size, image_size))
resize_img = resize_img / 255.0

plt.figure()
plt.imshow(resize_img)
plt.colorbar()
plt.grid(True)
plt.xlabel(class_names[0])
plt.show()

# Add the image to a batch where it's the only member.
resize_img = (np.expand_dims(resize_img,0))

print(resize_img.shape)

predictions_single = model.predict(resize_img)
predictions_label = class_names[np.argmax(predictions_single[0])]

print(predictions_single)
print(predictions_label)
