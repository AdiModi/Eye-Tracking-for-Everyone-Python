from matplotlib import pyplot as plt
from scipy import io
import project_path as pp
import os
import sys

right_eye_mean_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, 'mean_right_224.mat')
left_eye_mean_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, 'mean_left_224.mat')
face_mean_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, 'mean_face_224.mat')

# Checking whether the assigned .mat file paths exist
if not (os.path.exists(right_eye_mean_file_path) and os.path.exists(left_eye_mean_file_path) and os.path.exists(face_mean_file_path)):
    print('Mean File Paths (.mat files) Does not Exist!')
    print('Quitting...')
    sys.exit(0)

# Reading image means
right_eye_mean = io.loadmat(right_eye_mean_file_path)['image_mean']
left_eye_mean = io.loadmat(left_eye_mean_file_path)['image_mean']
face_mean = io.loadmat(face_mean_file_path)['image_mean']

# Normalizing image means to [0, 1]
right_eye_mean = right_eye_mean / 255
left_eye_mean = left_eye_mean / 255
face_mean = face_mean / 255

plt.figure(num='Means', figsize=(30, 10))

plt.subplot(1, 3, 1)
plt.imshow(right_eye_mean, cmap='gray')
plt.title('Right Eye')

plt.subplot(1, 3, 2)
plt.imshow(left_eye_mean, cmap='gray')
plt.title('Left Eye')

plt.subplot(1, 3, 3)
plt.imshow(face_mean, cmap='gray')
plt.title('Face')

plt.show()