import cv2
import numpy as np
import mediapipe as mp

import DEFINITION_FACEMASK
import helper_code as helper

import time
import matplotlib.pyplot as plt
from OneEuroFilter import OneEuroFilter
import copy

# from mediapipe.python._framework_bindings import timestamp
# Timestamp = timestamp.Timestamp

example_video = 0  # "jitter_test.mp4"  # 0  # "vid.avi"

if example_video == 0:
    cap = cv2.VideoCapture(example_video, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap = cv2.VideoCapture(example_video)

# initialize moving average filter
window_size = 5
centroid_list = []

centroid_coords_forehead = []
centroid_coords_left_cheek = []
centroid_coords_right_cheek = []
centroid_coords_forehead_filtered = []
centroid_coords_left_cheek_filtered = []
centroid_coords_right_cheek_filtered = []
frame_cnt = 0

# https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.proto
# https://gery.casiez.net/1euro/
# smoothing_calc = mp.calculators.util.landmarks_smoothing_calculator_pb2
# smoothing_calc.LandmarksSmoothingCalculatorOptions.OneEuroFilter.frequency.DESCRIPTOR.default_value = 3       # .OneEuroFilter.beta.DESCRIPTOR.default_value
# smoothing_calc.LandmarksSmoothingCalculatorOptions.OneEuroFilter(frequency=3, min_cutoff=30, beta=300, derivate_cutoff=50)
# one_euro_filter = smoothing_calc.LandmarksSmoothingCalculatorOptions.OneEuroFilter(frequency=3, min_cutoff=30, beta=300, derivate_cutoff=50)  # mp.util.filtering.one_euro_filter

# one-euro-filter config
config = {
    # Frequency of incoming frames defined in frames per seconds.
    'freq': 30,  # 30,
    # Minimum cutoff frequency. Start by tuning this parameter while keeping `beta = 0` to reduce jittering to the desired level.
    # 1Hz (the default value) is a good starting point.
    'mincutoff': 1.0,  # 0.8,  # 1.0,
    # Cutoff slope. After `min_cutoff` is configured, start increasing `beta` value to reduce the lag introduced by the `min_cutoff`.
    # Find the desired balance between jittering and lag.
    'beta': 0.0,  # 0.1,
    # Cutoff frequency for derivate. It is set to 1Hz in the original algorithm, but can be tuned to further smooth the speed (i.e. derivate) on the object.
    'dcutoff': 1.0  # 1.0
}

f = OneEuroFilter(**config)

# source: https://gist.github.com/igorbasko01/c51980df0ce9a516c8bcc4ff8e039eb7
num_coordinates = 3  # x, y, z
num_landmarks = 478  # len(results.multi_face_landmarks[0].landmark)

# Use a nested list comprehension to create the 3D array of filters as each landmarks coordinate needs its own OneEuroFilter
filters = np.array([[OneEuroFilter(**config) for j in range(num_coordinates)] for i in range(num_landmarks)])

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        startTime = time.time()

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            if example_video == 0:
                continue
            else:
                break
        # show jitter with a static image
        # frame = cv2.imread('test.jpg')

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        img_h, img_w = frame.shape[:2]
        mask_face = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_eyes = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_mouth = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_eyebrows = np.zeros((img_h, img_w), dtype=np.uint8)

        if results.multi_face_landmarks:
            # First parameter is the value to filter
            # the second parameter is the current timestamp in seconds
            # filtered = f(2.1, 0)

            filtered_landmarks_list = copy.deepcopy(results.multi_face_landmarks[0].landmark)
            timestamp = time.time()

            for idx, landmark in enumerate(filtered_landmarks_list):
                filtered_landmarks_list[idx].x = filters[idx][0](landmark.x, timestamp)
                filtered_landmarks_list[idx].y = filters[idx][1](landmark.y, timestamp)
                filtered_landmarks_list[idx].z = filters[idx][2](landmark.z, timestamp)

            # landmark_results = [results.multi_face_landmarks[0].landmark, filtered_landmarks_list]
            # for result in landmark_results:

            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            mesh_points_filtered = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                             for p in filtered_landmarks_list])  # results.multi_face_landmarks[0].landmark])

            difference = mesh_points_filtered - mesh_points

            # define mesh points of each ROI
            mesh_points_forehead = [mesh_points[DEFINITION_FACEMASK.FOREHEAD_list]]
            mesh_points_left_cheek = [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.LEFT_CHEEK_LIST,
                                                                     landmarks=results.multi_face_landmarks[0].landmark,
                                                                     img_w=img_w, img_h=img_h)]
            mesh_points_right_cheek = [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.RIGHT_CHEEK_LIST,
                                                                      landmarks=results.multi_face_landmarks[0].landmark,
                                                                      img_w=img_w, img_h=img_h)]

            # define filtered mesh points of each ROI
            mesh_points_forehead_filtered = [mesh_points_filtered[DEFINITION_FACEMASK.FOREHEAD_list]]
            mesh_points_left_cheek_filtered = [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.LEFT_CHEEK_LIST,
                                                                              landmarks=filtered_landmarks_list,
                                                                              img_w=img_w, img_h=img_h)]
            mesh_points_right_cheek_filtered = [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.RIGHT_CHEEK_LIST,
                                                                               landmarks=filtered_landmarks_list,
                                                                               img_w=img_w, img_h=img_h)]

            # isolate single ROIs from frame
            # ROI forehead
            output_roi_forehead = helper.segment_roi(frame, mesh_points_forehead)
            # ROI left cheek
            output_roi_left_cheek = helper.segment_roi(frame, mesh_points_left_cheek)
            # ROI right cheek
            output_roi_right_cheek = helper.segment_roi(frame, mesh_points_right_cheek)

            # save locations of each ROI's centroid in a list for plotting the jitter
            cX_forehead, cY_forehead = helper.calc_centroids(output_roi_forehead)
            centroid_coords_forehead.append((frame_cnt, (cX_forehead, cY_forehead)))
            cX_left_cheek, cY_left_cheek = helper.calc_centroids(output_roi_left_cheek)
            centroid_coords_left_cheek.append((frame_cnt, (cX_left_cheek, cY_left_cheek)))
            cX_right_cheek, cY_right_cheek = helper.calc_centroids(output_roi_right_cheek)
            centroid_coords_right_cheek.append((frame_cnt, (cX_right_cheek, cY_right_cheek)))

            # filtered ROI's centroids
            # ROI forehead
            output_roi_forehead_filtered = helper.segment_roi(frame, mesh_points_forehead_filtered)
            cX_forehead_filtered, cY_forehead_filtered = helper.calc_centroids(output_roi_forehead_filtered)
            centroid_coords_forehead_filtered.append((frame_cnt, (cX_forehead_filtered, cY_forehead_filtered)))

            # used for x-axis of plot
            frame_cnt += 1

            # ROIs of total face
            # drawing on the mask
            cv2.fillPoly(mask_face, mesh_points_forehead, (255, 255, 255, cv2.LINE_AA))
            cv2.fillPoly(mask_face, mesh_points_left_cheek, (255, 255, 255, cv2.LINE_AA))
            cv2.fillPoly(mask_face, mesh_points_right_cheek, (255, 255, 255, cv2.LINE_AA))
            output_roi_face = cv2.copyTo(frame, mask_face)

            # drawing ROI on the frames
            cv2.polylines(frame, [mesh_points[DEFINITION_FACEMASK.FOREHEAD_list]], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.polylines(frame, [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.LEFT_CHEEK_LIST,
                                                                 landmarks=results.multi_face_landmarks[0].landmark,
                                                                 img_w=img_w, img_h=img_h)], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.polylines(frame, [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.RIGHT_CHEEK_LIST,
                                                                 landmarks=results.multi_face_landmarks[0].landmark,
                                                                 img_w=img_w, img_h=img_h)], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.polylines(frame, mesh_points_forehead, True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(frame, mesh_points_left_cheek, True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(frame, mesh_points_right_cheek, True, (0, 255, 0), 1, cv2.LINE_AA)

            # crop frame to square bounding box, centered at centroid between all ROIs
            x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates(output_roi_face, results)
            distance_max = max(x_max - x_min, y_max - y_min)
            cX, cY = helper.calc_centroids(output_roi_face)

            # crop frame to square bounding box using centroids
            output_roi_face = output_roi_face[int(cY - distance_max / 2):int(cY + distance_max / 2),
                              int(cX - distance_max / 2):int(cX + distance_max / 2)]
            # frame = frame[int(cY - distance_max / 2):int(cY + distance_max / 2),
            #               int(cX - distance_max / 2):int(cX + distance_max / 2)]

            '''
            # Calculate the running average with a window size of 5
            if len(centroid_list) >= window_size:
                # shift elements in list to the left, to keep window_size and prevent list getting too large
                centroid_list.pop(0)
                centroid_list.append((cX, cY))
                rolling_mean_x, rolling_mean_y = helper.moving_average(np.array(centroid_list), 5)
            else:
                centroid_list.append((cX, cY))
                rolling_mean_x, rolling_mean_y = cX, cY

            # crop frame to square bounding box using rolling mean
            output_roi_face = output_roi_face[int(rolling_mean_y - distance_max / 2):int(rolling_mean_y + distance_max / 2),
                                              int(rolling_mean_x - distance_max / 2):int(rolling_mean_x + distance_max / 2)]
            '''

            # ToDo: wirft manchmal exception, wenn Gesicht aus dem Bild geht
            try:
                # crop ROI frames
                # ToDo: untersuche die Auswirkung von verschiedenen Interpolationen (INTER_AREA, INTER_CUBIC, INTER_LINEAR)
                # frame = cv2.resize(frame, (36, 36))
                # output_roi_face = cv2.resize(output_roi_face, (36, 36))
                output_roi_forehead = helper.resize_roi(output_roi_forehead, mesh_points_forehead)
                output_roi_left_cheek = helper.resize_roi(output_roi_left_cheek, mesh_points_left_cheek)
                output_roi_right_cheek = helper.resize_roi(output_roi_right_cheek, mesh_points_right_cheek)

                cv2.imshow('img', frame)
                # cv2.imshow('ROI face', output_roi_face)
                # cv2.imshow('ROI forehead', output_roi_forehead)
                # cv2.imshow('ROI left cheek', output_roi_left_cheek)
                # cv2.imshow('ROI right cheek', output_roi_right_cheek)
                # print(1/(time.time() - startTime))
            except cv2.error:
                pass

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# plot jitter of forehead centroid
# helper.plot_jitter(centroid_coords_forehead, "forehead")
# helper.plot_jitter(centroid_coords_left_cheek, "left_cheek")
# helper.plot_jitter(centroid_coords_right_cheek, "right_cheek")

# ToDo: interactive plot of parameter's effect according to page 16 in:
# https://www.math.uni-leipzig.de/~hellmund/Vorlesung/matplotlib19.pdf
helper.plot_jitter_comparison(centroid_coords_forehead, centroid_coords_forehead_filtered, "forehead")

plt.show()
