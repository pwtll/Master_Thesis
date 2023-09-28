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
from scipy.spatial import Delaunay

import matplotlib.colors as mcol
import matplotlib.cm as cm


@jit(nopython=True)
def calc_triangle_centroid_coordinates(index, triangle_centroid, centroid_coordinates, img_w, img_h, x_min, y_min):
    triangle_centroid[0] = int(triangle_centroid[0] * img_w - x_min)
    triangle_centroid[1] = int(triangle_centroid[1] * img_h - y_min)
    centroid_coordinates[index] = triangle_centroid[:2]


@jit(nopython=True)
def calc_barycentric_coords(pixel_coord, centroid_coordinates, vertices):
    A = np.column_stack((centroid_coordinates[vertices], np.ones(3)))
    A.astype("float64")
    b = np.array([pixel_coord[0], pixel_coord[1], 1], dtype="float64")

    barycentric_coords = np.linalg.solve(A.T, b)

    return barycentric_coords


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


def interpolate_surface_normal_angles(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max):
    # Perform Delaunay triangulation on the centroid coordinates.
    tri = Delaunay(centroid_coordinates)
    # Initialize an array to store the interpolated surface normal angles for each pixel.
    interpolated_surface_normal_angles = np.zeros(len(pixel_coordinates), dtype=np.float64)
    # Iterate through each pixel coordinate for interpolation.
    for i, pixel_coord in enumerate(pixel_coordinates):
        # Find the triangle that contains the current pixel using Delaunay triangulation.
        simplex_index = tri.find_simplex(pixel_coord)

        # ToDo: bei überlappenden Pixeln wegen zu großen Kopfdrehungen nur die niedrigeren Winkel nehmen
        if simplex_index != -1:
            # Get the vertices of the triangle that contains the pixel.
            vertices = tri.simplices[simplex_index]

            # Calculate the barycentric coordinates of the pixel within the triangle.
            barycentric_coords = calc_barycentric_coords(pixel_coord, centroid_coordinates, vertices)

            # Use the barycentric coordinates to interpolate the surface normal angle for the current pixel.
            interpolated_angle = sum(barycentric_coords[i] * surface_normal_angles[vertices[i]] for i in range(3))

            interpolated_surface_normal_angles[i] = interpolated_angle

    interpolated_surface_normal_angles = np.reshape(interpolated_surface_normal_angles, (-1, x_max - x_min))
    # interpolated_surface_normal_angles[interpolated_surface_normal_angles > 45] = 0
    # interpolated_surface_normal_angles[interpolated_surface_normal_angles == 0] = None  # 0  # None

    return interpolated_surface_normal_angles


def plot_interpolation_heatmap(interpolated_surface_normal_angles, xx, yy):
    # Make a user-defined colormap (source: https://stackoverflow.com/questions/25748183/python-making-color-bar-that-runs-from-red-to-blue)
    # cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "g", "r"])

    plt.imshow(interpolated_surface_normal_angles, cmap=cm1)  # , interpolation='nearest')    cmap='RdBu'    , cmap='seismic_r'
    create_colorbar(cm1, interpolated_surface_normal_angles)

    # plot contour lines each 15° between 0° to 90°
    CS = plt.contour(xx, yy, interpolated_surface_normal_angles, np.arange(90, step=30), colors="k", linewidths=0.75)
    plt.clabel(CS, inline=1, fontsize=10)

    # plt.show()
    plt.pause(.1)
    plt.draw()
    # plt.clf()


def create_colorbar(cm1, interpolated_surface_normal_angles):
    max_angle, min_angle, v = set_colorbar_ticks(interpolated_surface_normal_angles)

    # Make a normalizer that will map the angle values from [0,90] -> [0,1]
    cnorm = mcol.Normalize(vmin=min_angle, vmax=max_angle)
    # Turn these into an object that can be used to map time values to colors and can be passed to plt.colorbar()
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    plt.colorbar(cpick, label="Surface normal angle (°)", ticks=v)
    plt.axis('off')


@jit(nopython=True)
def set_colorbar_ticks(interpolated_surface_normal_angles):
    # get highest and lowest interpolated reflectance angles
    min_angle = np.nanmin(interpolated_surface_normal_angles)
    max_angle = np.nanmax(interpolated_surface_normal_angles)
    # alternative set of ticks
    # lin_start = round(min_angle+5, -1)
    # lin_stop = 10 * math.floor(max_angle / 10)
    # v = np.linspace(lin_start, lin_stop, int((lin_stop-lin_start)/10), endpoint=True)
    # v = np.append(min_angle, v)
    # v = np.append(v, max_angle)
    # set ticks for colorbar
    if min_angle < 15 - 5 and max_angle > 75 + 5:
        v = np.array([min_angle, 15, 30, 45, 60, 75, max_angle])
    elif min_angle < 15 - 5 and not max_angle > 75 + 5:
        v = np.array([min_angle, 15, 30, 45, 60, max_angle])
    elif not min_angle < 15 - 5 and max_angle > 75 + 5:
        v = np.array([min_angle, 30, 45, 60, 75, max_angle])
    elif not min_angle < 15 - 5 and not max_angle > 75 + 5:
        v = np.array([min_angle, 30, 45, 60, max_angle])
    return max_angle, min_angle, v



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

    # ToDo: untersuche Datensätze auf Frequenz der auftretenden Kopfrotationen, um Grenzfrequenz höher als die Rotationsfrequenz zu setzen,
    #  damit die Bewegung nicht tiefpass gefiltert wird
    # define lowpass filter with 2.9 Hz cutoff frequency
    b, a = iirfilter(20, Wn=2.9, fs=fps, btype="low", ftype="butter")

    for idx in np.arange(478):
        if len(landmark_coords_xyz_history[idx]) > 15:      # filter needs at least 15 values to work
            # apply filter forward and backward using filtfilt
            x_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[0] for coords_xyz in landmark_coords_xyz_history[idx]]))
            y_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[1] for coords_xyz in landmark_coords_xyz_history[idx]]))
            z_coords_lowpass_filtered = filtfilt(b, a, np.array([coords_xyz[2] for coords_xyz in landmark_coords_xyz_history[idx]]))

            landmark_coords_xyz_history[idx] = [(x_coords_lowpass_filtered[i], y_coords_lowpass_filtered[i], z_coords_lowpass_filtered[i]) for i in
                                                np.arange(0, len(x_coords_lowpass_filtered))]

    return np.array(landmark_coords_xyz_history, dtype=np.float64)


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
            - mesh_points_threshold_roi_ (list): List of sets of coordinates defining triangles within the optimal ROI that meet the angle threshold criteria and,
                                               if flag is set, are part of the optimal ROI.
            - mesh_points_outside_roi (list): List of coordinates defining triangles outside of the optimal ROI.
            - mask_threshold_roi (numpy.ndarray): A binary image mask indicating the optimal ROI.
            - mask_outside_roi (numpy.ndarray): A binary image mask indicating the area outside the optimal ROI.

    A list of sets of coordinates representing the triangles that meet the angle threshold criteria and, if flag is set, are part of the adaptive ROI.
    """
    global last_frames

    img_h, img_w = image.shape[:2]

    mesh_points_optimal_roi_ = []
    mask_forehead_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_left_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_right_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_outside_roi = np.zeros((img_h, img_w), dtype=np.uint8)

    angle_degrees_dict = {}

    # define landmarks to be plotted
    if constrain_roi:
        forehead_roi = DEFINITION_FACEMASK.FOREHEAD_TESSELATION_LARGE
        left_cheek_roi = DEFINITION_FACEMASK.LEFT_CHEEK_TESSELATION_LARGE
        right_cheek_roi = DEFINITION_FACEMASK.RIGHT_CHEEK_TESSELATION_LARGE

    if video_file != 0:
        # Extract landmarks' xyz-coordinates from the detected face
        landmark_coords_xyz = np.zeros((len(landmark_coords_xyz_history), 3))
        for index, face_landmarks in enumerate(landmark_coords_xyz_history):
            landmark_coords_xyz[index] = [face_landmarks[video_frame_count][0], face_landmarks[video_frame_count][1], face_landmarks[video_frame_count][2]]
    else:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks' xyz-coordinates from the detected face
            landmark_coords_xyz = np.zeros((len(face_landmarks.landmark), 3))
            for index, landmark in enumerate(face_landmarks.landmark):
                landmark_coords_xyz[index] = [landmark.x, landmark.y, landmark.z]

    # initialization for surface angle interpolation for all face pixels
    x_min, x_max = int(landmark_coords_xyz[:, 0].min() * img_w), int(landmark_coords_xyz[:, 0].max() * img_w)
    y_min, y_max = int(landmark_coords_xyz[:, 1].min() * img_h), int(landmark_coords_xyz[:, 1].max() * img_h)
    xx, yy = np.meshgrid(np.arange(x_max - x_min), np.arange(y_max - y_min))

    # image pixel coordinates for which to interpolate surface normal angles, each row is [x, y], starting from [0, 0] -> [img_w, img_h]
    pixel_coordinates = np.column_stack((xx.ravel(), yy.ravel()))

    # [x, y] coordinates of the triangle centroids
    centroid_coordinates = np.zeros((len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION), 2), dtype=np.int32)
    # surface normal angles for each triangle centroid
    surface_normal_angles = np.zeros(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))


    # Calculate angles between camera and surface normal vectors for whole face mesh tessellation
    for index, triangle in enumerate(DEFINITION_FACEMASK.FACE_MESH_TESSELATION):
        # calculate reflectance angle in degree
        angle_degrees = helper_functions.calculate_surface_normal_angle(landmark_coords_xyz, triangle)

        surface_normal_angles[index] = angle_degrees
        triangle_centroid = np.mean(np.array([landmark_coords_xyz[i] for i in triangle]), axis=0)
        calc_triangle_centroid_coordinates(index, triangle_centroid, centroid_coordinates, img_w, img_h, x_min, y_min)

        if constrain_roi:
            if triangle in forehead_roi:
                triangle = np.array(triangle)
                if check_acceptance(index, angle_degrees, angle_history, threshold):  # angle_degrees < threshold:  # angle_degrees < threshold: ## check_acceptance(index, angle_degrees):
                    # Extract the coordinates of the three landmarks of the triangle
                    triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
                    mesh_points_optimal_roi_.append(triangle_coords)

                    cv2.fillConvexPoly(mask_forehead_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
            elif triangle in left_cheek_roi:
                triangle = np.array(triangle)
                if check_acceptance(index, angle_degrees, angle_history, threshold):  # angle_degrees < threshold:  # angle_degrees < threshold: ## check_acceptance(index, angle_degrees):
                    # Extract the coordinates of the three landmarks of the triangle
                    triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
                    mesh_points_optimal_roi_.append(triangle_coords)

                    cv2.fillConvexPoly(mask_left_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
            elif triangle in right_cheek_roi:
                triangle = np.array(triangle)
                if check_acceptance(index, angle_degrees, angle_history, threshold):  # angle_degrees < threshold:  # angle_degrees < threshold: ## check_acceptance(index, angle_degrees):
                    # Extract the coordinates of the three landmarks of the triangle
                    triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
                    mesh_points_optimal_roi_.append(triangle_coords)

                    cv2.fillConvexPoly(mask_right_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
            else:
                 angle_degrees_dict.update({str(triangle): angle_degrees})

        else:
            if check_acceptance(index, angle_degrees, angle_history, threshold): # angle_degrees < threshold:
                # Extract the coordinates of the three landmarks of the triangle
                triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)

                cv2.fillConvexPoly(mask_outside_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))

    if use_convex_hull:
        mask_forehead_roi = helper.apply_convex_hull(mask_forehead_roi)
        mask_left_cheek_roi = helper.apply_convex_hull(mask_left_cheek_roi)
        mask_right_cheek_roi = helper.apply_convex_hull(mask_right_cheek_roi)

    if constrain_roi and use_outside_roi:
        interpolated_surface_normal_angles = interpolate_surface_normal_angles(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max)

        # JUST FOR DEBUGGING: three lines below just for plotting heatmap
        # interpolated_surface_normal_angles[interpolated_surface_normal_angles == 0] = None  # 0  # None
        # plt.clf()
        # plot_interpolation_heatmap(interpolated_surface_normal_angles, xx, yy)

        # calculate pixel area of optimal_roi
        mask_optimal_roi = mask_forehead_roi + mask_left_cheek_roi + mask_right_cheek_roi
        # ToDO: docs schreiben
        mask_eyes = helper.mask_eyes_out(mask_outside_roi, landmark_coords_xyz)
        # extract smallest interpolation angles and create new mask only including pixels with the same amount as mask_optimal_roi
        mask_outside_roi = extract_mask_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, mask_eyes, x_min, y_min)

        # mask out eyes from isolated face image
        # mask_eyes = helper.mask_eyes_out(mask_outside_roi, landmark_coords_xyz)
        # mask_outside_roi = cv2.copyTo(mask_outside_roi, mask_eyes)

        # mask out mask_optimal_roi from mask_outside_roi, so that the masks don't overlap after applying convexHull
        # inv_mask_optimal_roi = cv2.bitwise_not(mask_optimal_roi)
        # mask_outside_roi = cv2.bitwise_and(mask_outside_roi, inv_mask_optimal_roi)

        cv2.imshow('mask_optimal_roi', mask_optimal_roi)
        cv2.imshow('mask_outside_roi', mask_outside_roi)

    # calculate triangles with lowest angles and nearest distance to the centroid between the visible optimal ROIs
    # if constrain_roi and use_outside_roi:
    #
    #     # mask_optimal_roi = mask_forehead_roi + mask_left_cheek_roi + mask_right_cheek_roi
    #     # calculate pixel area of optimal_roi
    #     optimal_roi_area = helper.count_pixel_area(mask_optimal_roi)
    #     outside_roi_area = 0
    #
    #     while outside_roi_area <= optimal_roi_area != 0:
    #         # fill a black mask with white triangles of low angles
    #         mask_outside_roi = cv2.fillConvexPoly(mask_outside_roi, nearest_triangle, (255, 255, 255, cv2.LINE_AA))

    '''
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

    mask_cheek_roi = mask_left_cheek_roi + mask_right_cheek_roi

    # Sort the triangles based on their nearest Euclidean distance to the target triangle with lowest angle in optimal ROI
    # sorted_triangles = sorted(low_angle_triangle_coords, key=lambda triangle: helper.euclidean_distance(target_triangle, triangle))

    # calculate target centroid inside largest contour of optimal ROIs
    target_coords = helper.calc_centroid_of_largest_contour(mask_cheek_roi)

    # calculate target centroid in between visible optimal ROIs, weighted by the areas of optimal ROIs
    # target_coords = helper.calc_centroid_between_roi(mask_threshold_roi)

    # Sort the triangles based on their nearest Euclidean distance to the target centroid in between visible optimal ROIs
    sorted_triangles = sorted(low_angle_triangle_coords, key=lambda triangle: distance.euclidean(target_coords, np.mean(triangle, axis=0)))

    # mask_optimal_roi = mask_forehead_roi + mask_left_cheek_roi + mask_right_cheek_roi
    # calculate pixel area of optimal_roi
    optimal_roi_area = helper.count_pixel_area(mask_optimal_roi)
    outside_roi_area = 0

    for nearest_triangle in sorted_triangles:  # low_angle_triangle_coords:  # sorted_triangles: # low_angle_triangle_coords:  # sorted_triangles:
        if outside_roi_area <= optimal_roi_area != 0:
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
            #     # mask out mask_threshold_roi from mask_outside_roi, so that the masks don't overlap after applying convexHull
            #     inv_mask_threshold_roi = cv2.bitwise_not(mask_threshold_roi)
            #     mask_outside_roi = cv2.bitwise_and(mask_outside_roi, inv_mask_threshold_roi)

            # Count the non-black pixels
            outside_roi_area = helper.count_pixel_area(mask_outside_roi)
        else:
            break
    '''



    return mesh_points_optimal_roi_, mask_forehead_roi, mask_left_cheek_roi, mask_right_cheek_roi, mask_outside_roi


# @jit(nopython=True)
def extract_mask_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, mask_eyes, x_min, y_min):
    mask_interpolated_angles = subtract_optimal_roi_from_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, x_min, y_min)

    # set all zero values to None, to find the actual lowest angles
    # mask_interpolated_angles[mask_interpolated_angles == 0] = None

    mask_eyes = cv2.bitwise_not(mask_eyes, mask_eyes)
    mask_interpolated_angles = np.clip(mask_interpolated_angles - mask_eyes, 0, 255)

    set_zeroes_to_none(mask_interpolated_angles)

    # find the indices of the smallest values with the same count of optimal_roi_area + mask_eyes_area
    area = helper.count_pixel_area(mask_optimal_roi)#  + helper.count_pixel_area(mask_eyes)
    indices_of_smallest = np.argpartition(mask_interpolated_angles.flatten(), area)[:area]

    # in the new array set the values at the found indices to their original values and then to 255 to create a mask
    rows, cols = np.unravel_index(indices_of_smallest, mask_interpolated_angles.shape)
    mask_outside_roi = np.zeros_like(mask_interpolated_angles)
    mask_outside_roi[rows, cols] = mask_interpolated_angles[rows, cols]
    mask_outside_roi = mask_outside_roi.astype(np.uint8)
    mask_outside_roi[mask_outside_roi > 0] = 255

    return mask_outside_roi


@jit(nopython=True)
def set_zeroes_to_none(mask_interpolated_angles):
    # Iterate through the rows and columns of the image
    for i in np.arange(len(mask_interpolated_angles)):
        for j in np.arange(len(mask_interpolated_angles[i])):
            if mask_interpolated_angles[i][j] == 0:
                mask_interpolated_angles[i][j] = 255  # None


@jit(nopython=True)
def subtract_optimal_roi_from_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, x_min, y_min):
    # Create the larger mask_interpolated_angles array filled with zeros
    mask_interpolated_angles = np.zeros((img_h, img_w), dtype=interpolated_surface_normal_angles.dtype)

    # Copy the smaller interpolated_surface_normal_angles array into the larger mask_interpolated_angles array at the specified coordinates
    mask_interpolated_angles[y_min:y_min + interpolated_surface_normal_angles.shape[0], x_min:x_min + interpolated_surface_normal_angles.shape[1]] = interpolated_surface_normal_angles

    # subtract optimal_roi from outside_roi
    mask_interpolated_angles = np.clip(mask_interpolated_angles - mask_optimal_roi, 0, 255)
    # mask_interpolated_angles[np.isnan(mask_interpolated_angles)] = 0
    return mask_interpolated_angles


def main(video_file=None, fps=30, threshold=90, use_convex_hull=True, constrain_roi=True, use_outside_roi=False):
    if video_file == 0 or video_file is None:
        cap = cv2.VideoCapture(video_file, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(video_file)

    mp_face_mesh = mp.solutions.face_mesh
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

            mask_roi = None

            if results.multi_face_landmarks:

                if video_frame_count % 1 == 0:

                    # define mesh points of each ROI if mesh triangles are below threshold
                    try:
                        mesh_points_optimal_roi_, mask_forehead_roi, mask_left_cheek_roi, mask_right_cheek_roi, mask_outside_roi = calculate_roi(results, frame, threshold=threshold, constrain_roi=constrain_roi, use_convex_hull=use_convex_hull, use_outside_roi=use_outside_roi)
                    except IndexError:
                        print(video_frame_count)

                    mask_roi = mask_forehead_roi + \
                               mask_left_cheek_roi + \
                               mask_right_cheek_roi + \
                               mask_outside_roi  # mask_roi_optimal + mask_roi_outside

                    mask_roi_optimal = mask_forehead_roi + mask_left_cheek_roi + mask_right_cheek_roi
                    print("mask_roi_optimal: \t" + str(np.count_nonzero(mask_roi_optimal)) + "\tmask_roi_outside: \t" + str(np.count_nonzero(mask_outside_roi))
                          + "\tdifference: \t" + str(np.count_nonzero(mask_outside_roi)-np.count_nonzero(mask_roi_optimal)))

                    output_roi_face = cv2.copyTo(frame, mask_roi)
                    cv2.polylines(output_roi_face, mesh_points_optimal_roi_, True, (0, 255, 0), 1, cv2.LINE_AA)

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
                # use last valid data, even when no face is present in the current frame
                if mask_roi is not None and mask_roi.any():
                    output_roi_face = cv2.copyTo(frame, mask_roi)

                    distance_max = max(x_max - x_min, y_max - y_min)
                    output_roi_face = output_roi_face[int((y_min + y_max - distance_max) / 2 - bb_offset):int((y_min + y_max + distance_max) / 2 + bb_offset),
                                      int((x_min + x_max - distance_max) / 2 - bb_offset):int((x_min + x_max + distance_max) / 2 + bb_offset)]

                    output_roi_face = cv2.resize(output_roi_face, (256, 256), cv2.INTER_AREA)
                    cv2.imshow('ROI face', output_roi_face)

            cv2.imshow('img', frame)
            video_frame_count += 1

            if cv2.waitKey(40) & 0xFF == 27:
                break

        # perform Kolmogorov-Smirnov test to check if nose tip angle is normal distributed
        print(scipy.stats.kstest(angle_history[4], 'norm'))
        # create histogram to visualize values in dataset
        plt.hist(angle_history[4], edgecolor='black', bins=20)
        plt.show()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = 0 # "../data/vids/angle_jitter_test_2.mp4" ##"../data/triangle_jitter/triangle_jitter_raw_frame_4.avi"  #0 # "../data/vids/angle_jitter_test.mp4"  # VIPL_motion_scenario_p100_v6_source2.avi"  # angle_jitter_test.mp4" # 0

    # threshold angle in degrees. If the calculated angle is below this threshold, the heatmap will be drawn on the image
    threshold = 45

    fps = 20
    use_convex_hull = True
    constrain_roi = True
    use_outside_roi = True

    video_frame_count = 0
    landmark_coords_xyz_history = [[] for _ in np.arange(478)]
    angle_history = np.array([np.zeros(10) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))])
    was_visible = [np.zeros(5, dtype=bool) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))]

    last_frames = deque([np.full((720, 1280), 0, dtype=np.uint8)] * 3)  # np.zeros((720, 1280), dtype=np.uint8)

    # Initialize variables for frame storage and averaging
    frame_buffer = []  # Store the past 5 frames
    average_frame = None  # Initialize the average frame as None

    main(video_file=video_file, fps=fps, threshold=threshold, use_convex_hull=use_convex_hull, constrain_roi=constrain_roi, use_outside_roi=use_outside_roi)
