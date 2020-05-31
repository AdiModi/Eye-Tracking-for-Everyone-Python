import os

project_folder_path = os.path.abspath('..')

data_folder_path = os.path.abspath(os.path.join(project_folder_path, 'data'))
generated_folder_path = os.path.abspath(os.path.join(project_folder_path, 'generated'))
trained_models_folder_path = os.path.abspath(os.path.join(project_folder_path, 'trained models'))
resrc_folder_path = os.path.abspath(os.path.join(project_folder_path, 'resrc'))
src_folder_path = os.path.abspath(os.path.join(project_folder_path, 'src'))


# Give folder path of GazeCapture Dataset here
gaze_capture_dataset_folder_path = os.path.join(data_folder_path, 'GazeCapture')

# Give metadata file path here
metadata_file_path = os.path.join(gaze_capture_dataset_folder_path, 'reference_metadata.mat')

# Mobile features folder path
mobile_features_folder_path = os.path.join(project_folder_path, 'mobile features')

# Features for Mobile
features_for_mobile = os.path.join(project_folder_path, 'features for mobile')
