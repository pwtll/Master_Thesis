import cv2
import mediapipe as mp
import numpy as np
import scipy.stats

import DEFINITION_FACEMASK
import helper_code as helper
import time

from roi_segmentation.helper_code import calc_triangle_centroid_coordinates, check_acceptance, interpolate_surface_normal_angles, \
    extract_mask_outside_roi
from surface_normal_vector import helper_functions

from scipy.signal import iirfilter, filtfilt
import matplotlib.pyplot as plt


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


def calculate_roi(results, image, threshold=90, roi_mode="optimal_roi", constrain_roi=True, use_convex_hull=True, use_outside_roi=False):
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
    img_h, img_w = image.shape[:2]

    mesh_points_forehead = []
    mesh_points_left_cheek = []
    mesh_points_right_cheek = []
    mask_forehead_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_left_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_right_cheek_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_outside_roi = np.zeros((img_h, img_w), dtype=np.uint8)

    # define tesselation triangles contained in each roi
    if constrain_roi:
        forehead_roi = DEFINITION_FACEMASK.FOREHEAD_TESSELATION_LARGE
        left_cheek_roi = DEFINITION_FACEMASK.LEFT_CHEEK_TESSELATION_LARGE
        right_cheek_roi = DEFINITION_FACEMASK.RIGHT_CHEEK_TESSELATION_LARGE

    if video_file != 0:
        # Extract filtered landmark xyz-coordinates from the detected face in video
        landmark_coords_xyz = np.zeros((len(landmark_coords_xyz_history), 3))
        for index, face_landmarks in enumerate(landmark_coords_xyz_history):
            landmark_coords_xyz[index] = [face_landmarks[video_frame_count][0], face_landmarks[video_frame_count][1], face_landmarks[video_frame_count][2]]
    else:
        # Extract landmarks' xyz-coordinates from the detected face in realtime webcam input
        for face_landmarks in results.multi_face_landmarks:
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

        # calculate centroid coordinates of each triangle
        triangle_centroid = np.mean(np.array([landmark_coords_xyz[i] for i in triangle]), axis=0)
        centroid_coordinates[index] = calc_triangle_centroid_coordinates(triangle_centroid, img_w, img_h, x_min, y_min)

        # for interpolation, the reflectance angle is calculated for each triangle and mapped to the triangles centroid
        surface_normal_angles[index] = angle_degrees

        # check acceptance of triangle to be below threshold and add it to the ROI mask
        if constrain_roi:
            # Extract the coordinates of the three landmarks of the triangle
            triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)

            if triangle in forehead_roi:
                mesh_points_forehead.append(triangle_coords)         # necessary for bounding box
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_forehead_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
            elif triangle in left_cheek_roi:
                mesh_points_left_cheek.append(triangle_coords)      # necessary for bounding box
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_left_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
            elif triangle in right_cheek_roi:
                mesh_points_right_cheek.append(triangle_coords)     # necessary for bounding box
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_right_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
        else:
            # Extract the coordinates of the three landmarks of the triangle
            triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
            if check_acceptance(index, angle_degrees, angle_history, threshold):
                cv2.fillConvexPoly(mask_outside_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))

    if use_convex_hull:
        mask_forehead_roi = helper.apply_convex_hull(mask_forehead_roi)
        mask_left_cheek_roi = helper.apply_convex_hull(mask_left_cheek_roi)
        mask_right_cheek_roi = helper.apply_convex_hull(mask_right_cheek_roi)

    # calculate pixel area of optimal_roi
    if roi_mode == "optimal_roi":
        mask_optimal_roi = mask_forehead_roi + mask_left_cheek_roi + mask_right_cheek_roi
        mesh_points_bounding_box_ = mesh_points_forehead + mesh_points_left_cheek + mesh_points_right_cheek
    elif roi_mode == "forehead":
        mask_optimal_roi = mask_forehead_roi
        mesh_points_bounding_box_ = mesh_points_forehead
    elif roi_mode == "left_cheek":
        mask_optimal_roi = mask_left_cheek_roi
        mesh_points_bounding_box_ = mesh_points_left_cheek
    elif roi_mode == "right_cheek":
        mask_optimal_roi = mask_right_cheek_roi
        mesh_points_bounding_box_ = mesh_points_right_cheek
    else:
        raise Exception("No valid roi_mode selected. Valid roi_mode are: 'optimal_roi', 'forehead', 'left_cheek', 'right_cheek'.")

    if constrain_roi and use_outside_roi:
        interpolated_surface_normal_angles = interpolate_surface_normal_angles(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max)

        # JUST FOR DEBUGGING: three lines below just for plotting heatmap
        # interpolated_surface_normal_angles[interpolated_surface_normal_angles == 0] = None  # 0  # None
        # plt.clf()
        # plot_interpolation_heatmap(interpolated_surface_normal_angles, xx, yy)

        mask_eyes = helper.mask_eyes_out(mask_outside_roi, landmark_coords_xyz)
        # extract smallest interpolation angles and create new mask only including pixels with the same amount as mask_optimal_roi
        mask_outside_roi = extract_mask_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, mask_eyes, x_min, y_min)

        # cv2.imshow('mask_optimal_roi', mask_optimal_roi)
        # cv2.imshow('mask_outside_roi', mask_outside_roi)

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

    return mesh_points_bounding_box_, mask_optimal_roi, mask_outside_roi


def main(video_file=None, fps=30, threshold=90, roi_mode="optimal_roi", use_convex_hull=True, constrain_roi=True, use_outside_roi=False):
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
                    start_time = time.time()

                    # define mesh points of each ROI if mesh triangles are below threshold
                    try:
                        mesh_points_bounding_box_, mask_optimal_roi, mask_outside_roi = calculate_roi(results, frame,
                                                                                                      threshold=threshold,
                                                                                                      roi_mode=roi_mode,
                                                                                                      constrain_roi=constrain_roi,
                                                                                                      use_convex_hull=use_convex_hull,
                                                                                                      use_outside_roi=use_outside_roi)
                    except IndexError:
                        print(video_frame_count)

                    print(time.time() - start_time)

                    mask_roi = mask_outside_roi if use_outside_roi else mask_optimal_roi + mask_outside_roi

                    # print("mask_roi_optimal: \t" + str(np.count_nonzero(mask_roi_optimal)) + "\tmask_roi_outside: \t" + str(np.count_nonzero(mask_outside_roi))
                    #       + "\tdifference: \t" + str(np.count_nonzero(mask_outside_roi)-np.count_nonzero(mask_roi_optimal)))

                    output_roi_face = cv2.copyTo(frame, mask_roi)
                    cv2.polylines(output_roi_face, mesh_points_bounding_box_, True, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.polylines(frame, mesh_points_bounding_box_, True, (0, 255, 0), 1, cv2.LINE_AA)

                    # crop frame to square bounding box. The margins are either the outermost mesh point coordinates or (filtered) landmark coordinates
                    # processing video
                    if video_file != 0:
                        # use interpolated pixel ROI
                        if use_outside_roi:
                            # use filtered landmarks for a smoothed bounding box when using outside ROI during video processing
                            x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_filtered(output_roi_face, landmark_coords_xyz_history, video_frame_count)
                        else:
                            if constrain_roi:
                                # Use outermost coordinates of mesh points of the active ROI
                                x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_mesh_points(np.array(mesh_points_bounding_box_))
                            else:
                                # Use filtered landmarks for a smoothed bounding box of the whole face during video processing
                                x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_filtered(output_roi_face, landmark_coords_xyz_history, video_frame_count)
                    # processing real time webcam recording
                    else:
                        # Use outermost coordinates of mediapipe landmarks to create a bounding box during real time webcam recording
                        x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates(output_roi_face, results)

                    bb_offset = 2    # apply offset to the borders of bounding box
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
    # whether convexHull is applied to constrained ROI
    use_convex_hull = True
    # whether to constrain the ROI selection to forehead, left_cheek, right_cheek or all three combined (optimal_roi).
    # If False: take tesselation triangles which pass the test of acceptance (= below threshold or mean_angle < threshold + std_dev)
    constrain_roi = True
    # defines which ROI is extracted
    roi_mode = "optimal_roi"
    # defines whether only pixels outside of constrained ROI are included in extracted face region
    use_outside_roi = True

    video_frame_count = 0
    landmark_coords_xyz_history = [[] for _ in np.arange(478)]
    angle_history = np.array([np.zeros(10) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))])
    was_visible = [np.zeros(5, dtype=bool) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))]

    # last_frames = deque([np.full((720, 1280), 0, dtype=np.uint8)] * 3)  # np.zeros((720, 1280), dtype=np.uint8)

    # Initialize variables for frame storage and averaging
    # frame_buffer = []  # Store the past 5 frames
    # average_frame = None  # Initialize the average frame as None

    main(video_file=video_file, fps=fps, threshold=threshold, roi_mode=roi_mode, use_convex_hull=use_convex_hull, constrain_roi=constrain_roi, use_outside_roi=use_outside_roi)
