"""
This script iterates through a specified dataset and generates a UV-map plot illustrating the average reflectance
angles across the entire face tesselation of a single video.

The UV-maps for all videos in the dataset get saved in folder: data/dataset_tesselation_angles/NAME_OF_DATASET
    and keep the original dataset subfolder structure

The computation is performed by averaging angles for each frame in a video.

IMPORTANT:
It is necessary to run this script before executing: dataset_mean_angle_tesselation_heatmap.py
"""

import os
import time
import cv2
import mediapipe as mp
import numpy as np
from roi_segmentation.DEFINITION_FACEMASK import FACE_MESH_TESSELATION
import json
import helper_functions
from tqdm import tqdm
import concurrent.futures


def main(video_file=None, show_heatmap=False, threshold=90):
    mp_face_mesh = mp.solutions.face_mesh

    if video_file:
        # print("Processing: " + video_file)

        cap = cv2.VideoCapture(video_file)

        filepath = helper_functions.get_destination_path(video_file)
    else:
        example_video = 0  # "angle_test_short.mp4"
        if example_video == 0:
            cap = cv2.VideoCapture(example_video, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(example_video)

    uv_path = "uv_map.json"  # taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
    uv_map_dict = json.load(open(uv_path))
    uv_map = np.array([(uv_map_dict["u"][str(i)], uv_map_dict["v"][str(i)]) for i in range(468)])

    tesselation_mean_angles = {}

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # Extract landmarks' xyz-coordinates from the detected face
                    landmark_coords_xyz = []
                    for landmark in face_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        landmark_coords_xyz.append([x, y, z])

                    # Calculate angles between camera and surface normal vectors for whole face mesh tessellation and draw an angle heatmap
                    angle_dict = {}         # to save angle of each mesh triangle in a dict
                    for triangle in FACE_MESH_TESSELATION:
                        # calculate reflectance angle in degree
                        angle_degrees = helper_functions.calculate_surface_normal_angle(landmark_coords_xyz, triangle)

                        # display angle heatmap
                        if show_heatmap:
                            helper_functions.show_reflectance_angle_tesselation(image, landmark_coords_xyz, triangle, angle_degrees, threshold=threshold)

                        # save angle of each tesselation triangle in a dictionary
                        angle_dict.update({str(triangle): angle_degrees})

                        # compute online mean value of all past angles of each tesselation triangle in a dictionary
                        if str(triangle) in tesselation_mean_angles:
                            (count, mean, M2) = helper_functions.update_mean(tesselation_mean_angles[str(triangle)], angle_degrees)
                        else:
                            (count, mean, M2) = helper_functions.update_mean((0, 0, 0), angle_degrees)
                        tesselation_mean_angles.update({str(triangle): (count, mean, M2)})

            if show_heatmap:
                cv2.imshow("Face Mesh Tesselation", image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        # check if dictionary of reflectance angles is not empty. Occurs of no face is recognized by mediapipe
        if bool(tesselation_mean_angles):
            # Retrieve the mean, variance and sample variance from aggregated angles
            tesselation_angle_metrics = tesselation_mean_angles.copy()
            for triangle in FACE_MESH_TESSELATION:
                tesselation_angle_metrics[str(triangle)] = helper_functions.finalize(tesselation_angle_metrics[str(triangle)])

            # plot a heatmap of the mean reflectance angles of the face tesselation using UV coordinates
            mean_angle_heatmap_uv = helper_functions.plot_mean_angle_heatmap_uv(tesselation_angle_metrics, uv_map, show_heatmap=show_heatmap)

            if video_file:
                # save tesselation_angle_metrics dictionary in a pickle file
                helper_functions.pickle_dump_tesselation_angles(tesselation_angle_metrics, filepath=filepath)

                img_path_svg = os.path.splitext(filepath)[0] + ".svg"
                img_path_pdf = os.path.splitext(filepath)[0] + ".pdf"
                img_path_png = os.path.splitext(filepath)[0] + ".png"
                # save mean_angle_heatmap_uv plot in the same directory as tesselation_mean_angles
                mean_angle_heatmap_uv.figure.savefig(img_path_svg, dpi=600)
                mean_angle_heatmap_uv.figure.savefig(img_path_pdf, dpi=600)
                mean_angle_heatmap_uv.figure.savefig(img_path_png, dpi=600)

                # clear the current figure for next iteration
                helper_functions.plt.clf()
        else:
            raise Exception("No face detected. Please check video input.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # threshold angle in degrees. If the calculated angle is below this threshold, the heatmap will be drawn on the image
    threshold = 90

    # set True for real time webcam plot. set False to analyze the mean angles of a whole dataset saved in folder_path below
    real_time = False

    if real_time:
        main(video_file=None, show_heatmap=True, threshold=threshold)
    else:
        folder_path = 'g:/Uni/_Master/Semester 9 (Master Thesis)/Datasets/VIPL-HR-V1/'
        video_paths = helper_functions.get_video_paths_in_folder(folder_path)

        scenario = "v2\\source2"
        video_paths = [video_paths[i] for i in range(len(video_paths)) if scenario in video_paths[i].split("/")[-1]]

        start_time_dataset = time.time()

        # construct a pool of parallel processes for each CPU thread to reduce total computation time
        with concurrent.futures.ProcessPoolExecutor() as procs:
            list(tqdm(procs.map(main, [file for file in video_paths]), total=len(video_paths)))

        end_time_dataset = time.time()
        print("Runtime for whole dataset in seconds: " + str(end_time_dataset - start_time_dataset))
