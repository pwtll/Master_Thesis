import os
import time
import cv2
import mediapipe as mp
import numpy as np
import pickle
from roi_segmentation.DEFINITION_FACEMASK import FACE_MESH_TESSELATION
import json
import helper_functions


def main(dataset_path):
    # file_paths = helper_functions.get_video_paths_in_folder(folder_path)
    # filepath = helper_functions.get_destination_path(processed_dataset_path)

    uv_path = "uv_map.json"  # taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
    uv_map_dict = json.load(open(uv_path))
    uv_map = np.array([(uv_map_dict["u"][str(i)], uv_map_dict["v"][str(i)]) for i in range(468)])

    tesselation_metrics = {}

    # ToDo: iterate over dataset folder and compute an overall mean value of the reflectance angles of each triangle.
    file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith(".pkl"):
                file_paths.append(os.path.join(root, name))
                print(os.path.join(root, name))

    for file in file_paths:
        with open(file, 'rb') as f:
            tesselation_angle_metrics = pickle.load(f)

            for triangle in FACE_MESH_TESSELATION:
                # compute online mean value of all past angles of each tesselation triangle in a dictionary
                if str(triangle) in tesselation_metrics:
                    (count, mean, M2) = helper_functions.update_mean(tesselation_metrics[str(triangle)], tesselation_angle_metrics[str(triangle)][0])

                    if mean > 90:
                        print("Error")
                else:
                    (count, mean, M2) = helper_functions.update_mean((0, 0, 0), tesselation_angle_metrics[str(triangle)][0])
                tesselation_metrics.update({str(triangle): (count, mean, M2)})

    angle_dict = {}

    # Retrieve the mean, variance and sample variance of reflectance angles in pickle file
    for triangle in FACE_MESH_TESSELATION:
        tesselation_angle_metrics[str(triangle)] = helper_functions.finalize(tesselation_metrics[str(triangle)])
        # save mean angle of each tesselation triangle in a dictionary
        # angle_dict.update({str(triangle): tesselation_angle_metrics[str(triangle)][0]})

    # plot a heatmap of the mean reflectance angles of the face tesselation using UV coordinates
    mean_angle_heatmap_uv = helper_functions.plot_mean_angle_heatmap_uv(tesselation_angle_metrics, uv_map, show_heatmap=True)


if __name__ == "__main__":
    start_time_dataset = time.time()

    processed_dataset_path = '../data/dataset_tesselation_angles/VIPL-HR-V1/'

    main(processed_dataset_path)

    end_time_dataset = time.time()
    print("Runtime for whole dataset in seconds: " + str(end_time_dataset - start_time_dataset))
