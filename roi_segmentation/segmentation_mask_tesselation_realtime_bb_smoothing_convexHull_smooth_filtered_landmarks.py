import math
import os
import typing
from datetime import datetime
import csv

import cv2
import mediapipe as mp
import numpy as np
import scipy.stats

import DEFINITION_FACEMASK
import helper_code as helper
import time
from surface_normal_vector import helper_functions

from collections import deque
from scipy.spatial import distance
from scipy.signal import butter, lfilter, freqz, iirfilter, filtfilt
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def check_acceptance(index, angle_degrees, angle_history, threshold=90):
    """
    This function checks whether a triangle with a given angle is accepted based on its previous angle data.
    It calculates the mean angle and standard deviation of its previous angle values and checks if:
    1. current angle is within one standard deviation of the mean
    2. the triangle has appeared during the last 5 frames
    3. the mean angle during the last 5 frames is below the angle threshold + one standard deviation of the mean (last_5_mean < threshold + last_5_std_dev)

    :param index: (int): The index corresponding to the triangle in the DEFINITION_FACEMASK.FACE_MESH_TESSELATION
    :param angle_degrees: (float): The surface reflectance angle of the current triangle in degrees.
    :param threshold:(int, optional, default=90): The angle threshold in degrees. Triangles with angles below this threshold will be included in the adaptive ROI.
    :return: bool: True if the triangle is accepted, False otherwise.
    """

    angle_history[index] = np.roll(angle_history[index], shift=-1)  # Shift the values to make room for the new angle
    angle_history[index][-1] = angle_degrees  # Store the new angle in the array
    mean_angle = np.mean(angle_history[index])
    std_dev = np.std(angle_history[index])

    # Check if there are no zero values in angle_history[index] (occurs at first initialization)
    # and count how many past angle values are less than threshold + std_dev
    # ToDo: verbessere past_appearance, sodass es die gleiche boolsche bedingung hat, wie die if-Abfrage zum Akzeptieren der Dreiecke
    if np.count_nonzero(angle_history[index] == 0) == 0:
        past_appearance = angle_history[index] < threshold + std_dev
        # or (np.count_nonzero(angle_history[index][:-1] == 0) == 0 and np.mean(angle_history[index][:-1]) < threshold + np.std(angle_history[index][:-1]))
        past_appearance_count = np.count_nonzero(past_appearance)
    else:
        past_appearance_count = 0

    # accept triangle:
    # if its angle is below threshold,
    # or if it already appeared during the last 5 frames,
    # or if its mean angle during the last 5 frames is below threshold
    if angle_degrees < threshold \
            or past_appearance_count > 0 \
            or (np.count_nonzero(angle_history[index] == 0) == 0 and mean_angle < threshold + std_dev):
        return True
    return False


def calculate_roi(results, image, threshold=90, constrain_roi=True, use_convex_hull=True, use_outside_roi=False):
    """
     Calculates and extracts the region of interest (ROI) from an input image based on facial landmarks and their angles with respect to camera and surface normals.
     It uses the results from facial landmark detection to identify and extract triangles from the face mesh that fall below the specified angle threshold.
     The extracted triangles are returned as a list of sets of coordinates, which define the adaptive ROI.
     Additionally the function returns the binary mask images of the optimal ROI and the ROI outside of the optimal region.

    :param results: The results of facial landmark detection by mediapipe, containing facial landmarks detected in the image.
    :param image: The input image the facial landmarks were detected in and the adaptive ROI is to be calculated for.
    :param threshold:(int, optional, default=90): The angle threshold in degrees. Triangles with angles below this threshold will be included in the adaptive ROI.
    :param constrain_roi:(bool, optional, default=True): A flag indicating whether to constrain the adaptive ROI to a predefined set of optimal regions.
                                    If set to True, only triangles within the predefined regions will be considered.
    :param use_outside_roi: (bool, optional, default=False): If True, calculate ROIs outside of the constrained ROI.
    :return: tuple: A tuple containing the following elements:
            - mesh_points_optimal_roi_ (list): List of sets of coordinates defining triangles within the optimal ROI that meet the angle threshold criteria and,
                                               if flag is set, are part of the optimal ROI.
            - mesh_points_outside_roi (list): List of coordinates defining triangles outside of the optimal ROI.
            - mask_optimal_roi (numpy.ndarray): A binary image mask indicating the optimal ROI.
            - mask_outside_roi (numpy.ndarray): A binary image mask indicating the area outside the optimal ROI.

    A list of sets of coordinates representing the triangles that meet the angle threshold criteria and, if flag is set, are part of the adaptive ROI.
    """
    mesh_points_optimal_roi_ = []
    mesh_points_outside_roi_ = []
    global last_frames

    img_h, img_w = image.shape[:2]
    mask_optimal_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_outside_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    angle_degrees_dict = {}

    # define landmarks to be plotted
    if constrain_roi:
        optimal_rois = DEFINITION_FACEMASK.FOREHEAD_TESSELATION_LARGE \
                       + DEFINITION_FACEMASK.LEFT_CHEEK_TESSELATION_LARGE \
                       + DEFINITION_FACEMASK.RIGHT_CHEEK_TESSELATION_LARGE

    if video_file != 0:
        landmark_coords_xyz = []  # List()  # []
        for face_landmarks in landmark_coords_xyz_history:
            # Extract landmarks' xyz-coordinates from the detected face
            x, y, z = face_landmarks[video_frame_count][0], face_landmarks[video_frame_count][1], face_landmarks[video_frame_count][2]
            landmark_coords_xyz.append([x, y, z])
    else:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks' xyz-coordinates from the detected face
            landmark_coords_xyz = []
            for index, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                landmark_coords_xyz.append([x, y, z])

    landmark_coords_xyz = np.array(landmark_coords_xyz, dtype=np.float64)

    min_angle = float(180.0)   # float('inf')
    target_triangle = None

    # Calculate angles between camera and surface normal vectors for whole face mesh tessellation
    for index, triangle in enumerate(DEFINITION_FACEMASK.FACE_MESH_TESSELATION):
        if constrain_roi:
            if triangle in optimal_rois:
                triangle = np.array(triangle)

                # calculate reflectance angle in degree
                angle_degrees = helper_functions.calculate_surface_normal_angle(landmark_coords_xyz, triangle)

                # triangle_centroid = np.mean(np.array([landmark_coords_xyz[i] for i in triangle]), axis=0)

                if check_acceptance(index, angle_degrees, angle_history, threshold):  # angle_degrees < threshold:  # angle_degrees < threshold: ## check_acceptance(index, angle_degrees):
                    # Extract the coordinates of the three landmarks of the triangle
                    triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
                    mesh_points_optimal_roi_.append(triangle_coords)

                    cv2.fillConvexPoly(mask_optimal_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))

                    # Choose triangle with lowest angle as initial target triangle to calculate Euclidean distances for outside ROI
                    # if angle_degrees <= min_angle:
                    #     min_angle = angle_degrees
                    #     target_triangle = triangle_coords
            else:
                # calculate all reflectance angles outside of optimal roi
                angle_degrees = helper_functions.calculate_surface_normal_angle(landmark_coords_xyz, triangle)
                angle_degrees_dict.update({str(triangle): angle_degrees})

        else:
            # calculate reflectance angle in degree
            angle_degrees = helper_functions.calculate_surface_normal_angle(landmark_coords_xyz, triangle)

            if check_acceptance(index, angle_degrees, angle_history, threshold): # angle_degrees < threshold:
                # Extract the coordinates of the three landmarks of the triangle
                triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
                mesh_points_optimal_roi_.append(triangle_coords)

                cv2.fillConvexPoly(mask_optimal_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))

    if use_convex_hull:
        mask_optimal_roi = helper.apply_convex_hull(mask_optimal_roi)

        # if not use_outside_roi:         # else this is done later
        #     # mask out eyes from isolated face image
        #     mask_eyes = helper.mask_eyes_out(mask_optimal_roi, landmark_coords_xyz)
        #     mask_optimal_roi = cv2.copyTo(mask_optimal_roi, mask_eyes)

    # calculate triangles with lowest angles and nearest distance to the centroid between the visible optimal ROIs
    if constrain_roi and use_outside_roi:
        # sort angle_degrees by ascending angles
        angles_list = sorted(list(angle_degrees_dict.values()))

        # get triangle coordinates of the triangles sorted by lowest angles
        low_angle_triangle_coords = []
        for low_angle in angles_list:  # angles_list[:200]:
            # get index of triangle in FACE_MESH_TESSELATION
            index = DEFINITION_FACEMASK.FACE_MESH_TESSELATION.index(helper_functions.get_triangle_indices_from_angle(angle_degrees_dict, low_angle))
            if check_acceptance(index, low_angle, angle_history, threshold):
                low_angle_triangle_coords.append(helper_functions.get_triangle_coords(image, landmark_coords_xyz,
                                                                                      helper_functions.get_triangle_indices_from_angle(angle_degrees_dict, low_angle)))

        # Sort the triangles based on their nearest Euclidean distance to the target triangle with lowest angle in optimal ROI
        # sorted_triangles = sorted(low_angle_triangle_coords, key=lambda triangle: helper.euclidean_distance(target_triangle, triangle))

        # calculate target centroid inside largest contour of optimal ROIs
        # target_coords = helper.calc_centroid_of_largest_contour(mask_optimal_roi)

        # calculate target centroid in between visible optimal ROIs, weighted by the areas of optimal ROIs
        # target_coords = helper.calc_centroid_between_roi(mask_optimal_roi)

        # Sort the triangles based on their nearest Euclidean distance to the target centroid in between visible optimal ROIs
        # sorted_triangles = sorted(low_angle_triangle_coords, key=lambda triangle: distance.euclidean(target_coords, np.mean(triangle, axis=0)))

        # calculate pixel area of optimal_roi
        optimal_roi_area = helper.count_pixel_area(mask_optimal_roi)
        outside_roi_area = 0

        for nearest_triangle in low_angle_triangle_coords:  # sorted_triangles: # low_angle_triangle_coords:  # sorted_triangles:
            if outside_roi_area <= optimal_roi_area != 0:
                # ToDo: Robustheit auf outside ROI anwenden
                # calculate reflectance angle in degree
                # angle_degrees = helper_functions.calculate_surface_normal_angle(landmark_coords_xyz, nearest_triangle)
                # get index of triangle in FACE_MESH_TESSELATION
                # index = DEFINITION_FACEMASK.FACE_MESH_TESSELATION.index(helper_functions.get_triangle_indices_from_angle(angle_degrees_dict, angle_degrees))

                # if check_acceptance(index, angle_degrees, angle_history, threshold):

                mesh_points_outside_roi_.append(nearest_triangle)

                # fill a black mask with white triangles of low angles
                mask_outside_roi = cv2.fillConvexPoly(mask_outside_roi, nearest_triangle, (255, 255, 255, cv2.LINE_AA))

                # mask out eyes from isolated face image
                # if use_convex_hull:
                #     mask_outside_roi = helper.apply_convex_hull(mask_outside_roi)
                #
                #     # mask out eyes from isolated face image
                #     mask_eyes = helper.mask_eyes_out(mask_outside_roi, landmark_coords_xyz)
                #     mask_outside_roi = cv2.copyTo(mask_outside_roi, mask_eyes)
                #
                #     # mask out mask_optimal_roi from mask_outside_roi, so that the masks don't overlap after applying convexHull
                #     inv_mask_optimal_roi = cv2.bitwise_not(mask_optimal_roi)
                #     mask_outside_roi = cv2.bitwise_and(mask_outside_roi, inv_mask_optimal_roi)

                # Count the non-black pixels
                outside_roi_area = helper.count_pixel_area(mask_outside_roi)
            else:
                break

    return mesh_points_optimal_roi_, mesh_points_outside_roi_, mask_optimal_roi, mask_outside_roi


def low_pass_filter_landmarks(video_file):
    cap_filter = cv2.VideoCapture(video_file)
    mp_face_mesh = mp.solutions.face_mesh

    video_frame_count_filter = 0
    last_valid_coords = None  # Initialize a variable to store the last valid coordinates

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while cap_filter.isOpened():
            success, frame = cap_filter.read()
            if not success:
                print("Ignoring empty camera frame during landmark filtering: " + str(video_frame_count_filter))
                # If loading a video, use 'break' instead of 'continue'.
                if video_file == 0:
                    continue
                else:
                    break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # Extract landmarks' xyz-coordinates from the detected face
                    for index, landmark in enumerate(face_landmarks.landmark):
                        x, y, z = landmark.x, landmark.y, landmark.z
                        landmark_coords_xyz_history[index].append((x, y, z))
                        last_valid_coords = (x, y, z)
            else:
                # No face detected, append the last valid coordinates (if available)
                if last_valid_coords is not None:
                    for index in np.arange(len(landmark_coords_xyz_history)):
                        landmark_coords_xyz_history[index].append(last_valid_coords)

            video_frame_count_filter += 1

    cap_filter.release()

    # define lowpass filter with 2.5 Hz cutoff frequency
    b, a = iirfilter(20, Wn=2.5, fs=fps, btype="low", ftype="butter")

    for idx in np.arange(478):
        # x_coords = np.array([coords_xyz[0] for coords_xyz in landmark_coords_xyz_history[idx]])
        # y_coords = np.array([coords_xyz[1] for coords_xyz in landmark_coords_xyz_history[idx]])
        # z_coords = np.array([coords_xyz[2] for coords_xyz in landmark_coords_xyz_history[idx]])

        if len(landmark_coords_xyz_history[idx]) > 15:      # filter needs at least 15 values to work
            # apply filter forward and backward using filtfilt
            x_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[0] for coords_xyz in landmark_coords_xyz_history[idx]]))
            y_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[1] for coords_xyz in landmark_coords_xyz_history[idx]]))
            z_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[2] for coords_xyz in landmark_coords_xyz_history[idx]]))

            landmark_coords_xyz_history[idx] = [(x_coords_lowpass_filtered[i], y_coords_lowpass_filtered[i], z_coords_lowpass_filtered[i]) for i in
                                                np.arange(0, len(x_coords_lowpass_filtered))]

    return np.array(landmark_coords_xyz_history, dtype=np.float64)


def main(video_file=None, fps=30, threshold=90, use_convex_hull=True, constrain_roi=True, use_outside_roi=False):
    if video_file == 0 or video_file is None:
        cap = cv2.VideoCapture(video_file, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        video_name = "live_video"
    else:
        cap = cv2.VideoCapture(video_file)

        video_name = os.path.basename(video_file)

    mp_face_mesh = mp.solutions.face_mesh

    # Create a CSV file for storing pixel area data
    # csv_filename = f"../data/ROI_area_log/{os.path.splitext(video_name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    # csv_file = open(csv_filename, mode='w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['Frame Number', 'Total Pixel Area', 'Optimal Pixel Area', 'Outside Pixel Area', 'Difference'])

    global video_frame_count

    # writer_raw_frame = cv2.VideoWriter('../data/outside_roi_comparison/raw_frame_centroid_between_squared_1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))
    # writer_roi_mask = cv2.VideoWriter('../data/outside_roi_comparison/facemesh_smallest_triangle_angle_1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (256, 256))

    if video_file != 0:
        landmark_coords_xyz_history = low_pass_filter_landmarks(video_file)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame: " + str(video_frame_count))
                # If loading a video, use 'break' instead of 'continue'.
                if video_file == 0:
                    continue
                else:
                    break
            frame = cv2.flip(frame, 1)


            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:

                if video_frame_count % 1 == 0:

                    # define mesh points of each ROI if mesh triangles are below threshold
                    try:
                        mesh_points_apaptive_roi, mesh_points_outside_roi, mask_roi_optimal, mask_roi_outside = calculate_roi(results, frame, threshold=threshold, constrain_roi=constrain_roi, use_convex_hull=use_convex_hull, use_outside_roi=use_outside_roi)
                    except IndexError:
                        print(video_frame_count)

                    mask_roi = mask_roi_optimal + mask_roi_outside  # mask_roi_optimal + mask_roi_outside

                    # print("mask_roi_optimal: \t" + str(np.count_nonzero(mask_roi_optimal)) + "\tmask_roi_outside: \t" + str(np.count_nonzero(mask_roi_outside))
                    #       + "\tdifference: \t" + str(np.count_nonzero(mask_roi_outside)-np.count_nonzero(mask_roi_optimal)))

                    # csv_writer.writerow([video_frame_count, helper.count_pixel_area(mask_roi), helper.count_pixel_area(mask_roi_optimal), helper.count_pixel_area(mask_roi_outside),
                    #                      helper.count_pixel_area(mask_roi_outside) - helper.count_pixel_area(mask_roi_optimal)])

                    output_roi_face = cv2.copyTo(frame, mask_roi)
                    cv2.polylines(output_roi_face, mesh_points_apaptive_roi, True, (0, 255, 0), 1, cv2.LINE_AA)

                    # crop frame to square bounding box, centered at centroid between all ROIs
                    bb_offset = 10

                    if video_file == 0:
                        x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates(output_roi_face, results)
                    else:
                        x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_filtered(output_roi_face, landmark_coords_xyz_history, video_frame_count)
                    distance_max = max(x_max - x_min, y_max - y_min)
                    output_roi_face = output_roi_face[int((y_min+y_max-distance_max)/2-bb_offset):int((y_min+y_max+distance_max)/2+bb_offset),
                                                      int((x_min+x_max-distance_max)/2-bb_offset):int((x_min+x_max+distance_max)/2+bb_offset)]
                try:
                    output_roi_face = cv2.resize(output_roi_face, (256, 256), cv2.INTER_AREA)
                    cv2.imshow('ROI face', output_roi_face)
                    # writer_roi_mask.write(output_roi_face)
                except cv2.error:
                    print("cv2 Error: " + str(video_frame_count))
                    pass
            else:
                # use last valud data, even when no face is present in the current frame
                if mask_roi.any():
                    output_roi_face = cv2.copyTo(frame, mask_roi)

                    distance_max = max(x_max - x_min, y_max - y_min)
                    output_roi_face = output_roi_face[int((y_min + y_max - distance_max) / 2 - bb_offset):int((y_min + y_max + distance_max) / 2 + bb_offset),
                                      int((x_min + x_max - distance_max) / 2 - bb_offset):int((x_min + x_max + distance_max) / 2 + bb_offset)]

                    output_roi_face = cv2.resize(output_roi_face, (256, 256), cv2.INTER_AREA)
                    cv2.imshow('ROI face', output_roi_face)

            cv2.imshow('img', frame)
            video_frame_count += 1

            #writer_raw_frame.write(frame)

            if cv2.waitKey(40) & 0xFF == 27:
                break

        # perform Kolmogorov-Smirnov test to check if nose tip angle is normal distributed
        print(scipy.stats.kstest(angle_history[4], 'norm'))
        # create histogram to visualize values in dataset
        plt.hist(angle_history[4], edgecolor='black', bins=20)
        plt.show()

        #writer_raw_frame.release()
        #writer_roi_mask.release()

        cap.release()
        cv2.destroyAllWindows()
        # csv_file.close()


if __name__ == "__main__":
    video_file = 0 # "../data/vids/angle_jitter_test_2.mp4" ##"../data/triangle_jitter/triangle_jitter_raw_frame_4.avi"  #0 # "../data/vids/angle_jitter_test.mp4"  # VIPL_motion_scenario_p100_v6_source2.avi"  # angle_jitter_test.mp4" # 0

    # threshold angle in degrees. If the calculated angle is below this threshold, the heatmap will be drawn on the image
    threshold = 45

    fps = 20
    use_convex_hull = False
    constrain_roi = False
    use_outside_roi = False

    video_frame_count = 0
    landmark_coords_xyz_history = [[] for _ in np.arange(478)]
    angle_history = np.array([np.zeros(200) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))])
    was_visible = [np.zeros(5, dtype=bool) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))]

    last_frames = deque([np.full((720, 1280), 0, dtype=np.uint8)] * 3)  # np.zeros((720, 1280), dtype=np.uint8)

    # Initialize variables for frame storage and averaging
    frame_buffer = []  # Store the past 5 frames
    average_frame = None  # Initialize the average frame as None

    main(video_file=video_file, fps=fps, threshold=threshold, use_convex_hull=use_convex_hull, constrain_roi=constrain_roi, use_outside_roi=use_outside_roi)
