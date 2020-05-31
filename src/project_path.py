import os
import inspect

current_file_path = inspect.getfile(inspect.currentframe())
project_folder_path = os.path.abspath(os.path.join(current_file_path, '..', '..'))

src_folder_path = os.path.abspath(os.path.join(project_folder_path, 'src'))
data_folder_path = os.path.abspath(os.path.join(project_folder_path, 'data'))
generated_folder_path = os.path.abspath(os.path.join(project_folder_path, 'generated'))
trained_models_folder_path = os.path.abspath(os.path.join(project_folder_path, 'trained models'))

# Give folder path of GazeCapture Dataset here
gaze_capture_dataset_folder_path = os.path.join(data_folder_path, 'GazeCapture')

# Give metadata file path here
metadata_file_path = os.path.join(gaze_capture_dataset_folder_path, 'reference_metadata.mat')
