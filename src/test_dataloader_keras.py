import random
import numpy as np
from dataloader_keras import DataBatcher
from matplotlib import pyplot as plt

db = DataBatcher(type=DataBatcher.TEST)
inputs, outputs = db[0]

right_eye_images = inputs[0]
left_eye_images = inputs[1]
face_images = inputs[2]
face_grids = inputs[3]

plt.figure(num='Test', figsize=(20, 20))

idx = random.randint(0, right_eye_images.shape[0])

plt.subplot(2, 2, 1)
plt.imshow(right_eye_images[idx], cmap='gray')
plt.title('Right Eye')

plt.subplot(2, 2, 2)
plt.imshow(left_eye_images[idx], cmap='gray')
plt.title('Left Eye')

plt.subplot(2, 2, 3)
plt.imshow(face_images[idx], cmap='gray')
plt.title('Face')

plt.subplot(2, 2, 4)
plt.imshow(np.reshape(face_grids[idx], (25, 25)), cmap='gray')
plt.title('Face Grid')

plt.show()