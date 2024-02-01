"""
This script uses facial landmarks detected in a video file or live webcam feed to perform ROI segmentation
based on specified parameters.


The parameters are:

 -  constrain_roi: whether to constrain the ROI selection to the specified roi_mode or just apply the angle threshold

 -  threshold: threshold angle in degrees. If the calculated angle is below this threshold,
               the face triangle gets accepted into segmented facial region

 -  roi_mode: defines which ROI is extracted.
              Choose between: forehead, left_cheek, right_cheek or optimal_roi (= all three combined)

 -  use_convex_hull: whether convexHull is applied to constrained ROI. It's recommended to leave this as True

 -  use_outside_roi: defines whether only pixels outside of the specified roi_mode are included in extracted face region
                     The region outside of roi_mode will be interpolated for higher spatial resolution of reflectance
                     angles. The size of outside_roi will be the same, as the original ROI specified by roi_mode

"""


import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import iirfilter, filtfilt
import matplotlib.pyplot as plt

import DEFINITION_FACEMASK
import helper_code as helper
from roi_segmentation.helper_code import calc_triangle_centroid_coordinates, check_acceptance, interpolate_surface_normal_angles, \
    extract_mask_outside_roi
from surface_normal_vector import helper_functions


def low_pass_filter_landmarks(video_file):
    cap_filter = cv2.VideoCapture(video_file)
    mp_face_mesh = mp.solutions.face_mesh

    video_frame_count_filter = 0
    last_valid_coords = None  # Initialize a variable to store the last valid coordinates

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
    ) as face_mesh:
        while cap_filter.isOpened():
            success, frame = cap_filter.read()
            if not success:
                print("Ignoring empty camera frame during landmark filtering: " + str(video_frame_count_filter))
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

    # define lowpass filter with 3.5 Hz cutoff frequency
    b, a = iirfilter(20, Wn=3.5, fs=fps, btype="low", ftype="butter")

    for idx in np.arange(478):
        if len(landmark_coords_xyz_history[idx]) > 15:  # filter needs at least 15 values to work
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

    mean_angle_forehead = []
    mean_angle_left_cheek = []
    mean_angle_right_cheek = []


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


    # define tesselation triangles contained in each roi
    if constrain_roi:
        forehead_roi = DEFINITION_FACEMASK.FOREHEAD_TESSELATION_LARGE
        left_cheek_roi = DEFINITION_FACEMASK.LEFT_CHEEK_TESSELATION_LARGE
        right_cheek_roi = DEFINITION_FACEMASK.RIGHT_CHEEK_TESSELATION_LARGE

        if use_outside_roi:
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

        # check acceptance of triangle to be below threshold and add it to the ROI mask
        if constrain_roi:
            if use_outside_roi:
                # calculate centroid coordinates of each triangle
                triangle_centroid = np.mean(np.array([landmark_coords_xyz[i] for i in triangle]), axis=0)
                centroid_coordinates[index] = calc_triangle_centroid_coordinates(triangle_centroid, img_w, img_h, x_min,
                                                                                 y_min)

                # for interpolation, the reflectance angle is calculated for each triangle and mapped to the triangles centroid
                surface_normal_angles[index] = angle_degrees


            # Extract the coordinates of the three landmarks of the triangle
            triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)

            if triangle in forehead_roi:
                mesh_points_forehead.append(triangle_coords)  # necessary for bounding box
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_forehead_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
                    mean_angle_forehead.append(angle_degrees)
            elif triangle in left_cheek_roi:
                mesh_points_left_cheek.append(triangle_coords)  # necessary for bounding box
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_left_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
                    mean_angle_left_cheek.append(angle_degrees)
            elif triangle in right_cheek_roi:
                mesh_points_right_cheek.append(triangle_coords)  # necessary for bounding box
                if check_acceptance(index, angle_degrees, angle_history, threshold):
                    cv2.fillConvexPoly(mask_right_cheek_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))
                    mean_angle_right_cheek.append(angle_degrees)
        else:
            # Extract the coordinates of the three landmarks of the triangle
            triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
            if check_acceptance(index, angle_degrees, angle_history, threshold):
                cv2.fillConvexPoly(mask_outside_roi, triangle_coords, (255, 255, 255, cv2.LINE_AA))

    if use_convex_hull:
        mask_forehead_roi = helper.apply_convex_hull(mask_forehead_roi)
        mask_left_cheek_roi = helper.apply_convex_hull(mask_left_cheek_roi)
        mask_right_cheek_roi = helper.apply_convex_hull(mask_right_cheek_roi)

    # set mask of defined ROI (one of ["optimal_roi", "forehead", "left_cheek", "right_cheek"])
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
        interpolated_surface_normal_angles = helper.interpolate_surface_normal_angles_scipy(centroid_coordinates, pixel_coordinates, surface_normal_angles,
                                                                                            x_min, x_max)
        # JUST FOR DEBUGGING: three lines below just for plotting heatmap
        interpolated_surface_normal_angles[interpolated_surface_normal_angles == 0] = None  # 0  # None
        plt.figure(1)
        plt.clf()
        plt.title('Interpolated reflectance angles', fontsize=14) # , fontweight='bold')
        helper.plot_interpolation_heatmap(interpolated_surface_normal_angles, xx, yy)
        # plt.show()

        mask_eyes = helper.mask_eyes_out(mask_outside_roi, landmark_coords_xyz)
        # extract smallest interpolation angles and create new mask only including pixels with the same amount as mask_optimal_roi
        mask_outside_roi = extract_mask_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, mask_eyes, x_min, y_min)

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

    if video_file != 0:
        landmark_coords_xyz_history = low_pass_filter_landmarks(video_file)


    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame: " + str(video_frame_count))
                if video_file == 0:
                    continue
                else:
                    break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            mask_roi = None
            output_roi_face = None

            if results.multi_face_landmarks:

                if video_frame_count % 1 == 0:
                    # define mesh points of each ROI if mesh triangles are below threshold
                    try:
                        mesh_points_bounding_box_, mask_optimal_roi, mask_outside_roi = calculate_roi(results, frame,
                                                                                                      threshold=threshold,
                                                                                                      roi_mode=roi_mode,
                                                                                                      constrain_roi=constrain_roi,
                                                                                                      use_convex_hull=use_convex_hull,
                                                                                                      use_outside_roi=use_outside_roi)
                    except IndexError as ie:
                        print("IndexError: " + str(video_frame_count))

                    mask_roi = mask_outside_roi if use_outside_roi else mask_optimal_roi + mask_outside_roi
                    output_roi_face = cv2.copyTo(frame, mask_roi)


                    # crop frame to square bounding box. The margins are either the outermost mesh point coordinates or (filtered) landmark coordinates
                    # processing video
                    if video_file != 0:
                        if video_frame_count < landmark_coords_xyz_history.shape[1]:
                            # use interpolated pixel ROI
                            if use_outside_roi:
                                # use filtered landmarks for a smoothed bounding box when using outside ROI during video processing
                                x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_filtered(output_roi_face, landmark_coords_xyz_history,
                                                                                                          video_frame_count)
                            else:
                                if constrain_roi:
                                    # Use outermost coordinates of mesh points of the active ROI
                                    x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_mesh_points(np.array(mesh_points_bounding_box_))
                                else:
                                    # Use filtered landmarks for a smoothed bounding box of the whole face during video processing
                                    x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_filtered(output_roi_face, landmark_coords_xyz_history,
                                                                                                              video_frame_count)
                        else:
                            pass
                    # processing real time webcam recording
                    else:
                        if roi_mode == "optimal_roi":
                            # Use outermost coordinates of mediapipe landmarks to create a bounding box during real time webcam recording
                            x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates(output_roi_face, results)
                        else:
                            # Use outermost coordinates of mesh points of the active ROI
                            x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates_mesh_points(
                                                                                    np.array(mesh_points_bounding_box_))

                    bb_offset = 2  # apply offset to the borders of bounding box
                    output_roi_face, x_max_bb, x_min_bb, y_max_bb, y_min_bb = helper.apply_bounding_box(output_roi_face,
                                                                                                        bb_offset,
                                                                                                        x_min, y_min,
                                                                                                        x_max, y_max)
                    output_mask, x_max_bb, x_min_bb, y_max_bb, y_min_bb = helper.apply_bounding_box(mask_roi,
                                                                                                        bb_offset,
                                                                                                        x_min, y_min,
                                                                                                        x_max, y_max)

                try:
                    output_roi_face = cv2.resize(output_roi_face, (256, 256), cv2.INTER_AREA)
                    cv2.imshow('ROI face', output_roi_face)
                except cv2.error as cv_e:
                    print("cv2 Error resize: " + str(video_frame_count))
                    print(str(cv_e))
                    pass
            else:
                # use last valid data, even when no face is present in the current frame
                if mask_roi is not None and mask_roi.any():
                    output_roi_face = cv2.copyTo(frame, mask_roi)

                    output_roi_face = output_roi_face[int(y_min_bb):int(y_max_bb), int(x_min_bb):int(x_max_bb)]

                    output_roi_face = cv2.resize(output_roi_face, (256, 256), cv2.INTER_AREA)
                    cv2.imshow('ROI face', output_roi_face)

            cv2.imshow('img', frame)
            video_frame_count += 1

            if cv2.waitKey(2) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = 0      # choose 0 for live webcam video or define filepath of video file

    # threshold angle in degrees. If the calculated angle is below this threshold, the heatmap will be drawn on the image
    threshold = 90

    fps = 30
    # whether convexHull is applied to constrained ROI
    use_convex_hull = True
    # whether to constrain the ROI selection to forehead, left_cheek, right_cheek or all three combined (optimal_roi).
    # If False: take tesselation triangles which pass the test of acceptance (= below threshold or mean_angle < threshold + std_dev)
    constrain_roi = True
    # defines which ROI is extracted
    roi_mode = "optimal_roi"  # choose between [ "optimal_roi", "forehead", "left_cheek", "right_cheek"]
    # defines whether only pixels outside of constrained ROI are included in extracted face region
    use_outside_roi = True

    video_frame_count = 0
    landmark_coords_xyz_history = [[] for _ in np.arange(478)]
    angle_history = np.array([np.zeros(10) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))])
    was_visible = [np.zeros(5, dtype=bool) for _ in np.arange(len(DEFINITION_FACEMASK.FACE_MESH_TESSELATION))]

    main(video_file=video_file,
         fps=fps,
         threshold=threshold,
         roi_mode=roi_mode,
         use_convex_hull=use_convex_hull,
         constrain_roi=constrain_roi,
         use_outside_roi=use_outside_roi)
