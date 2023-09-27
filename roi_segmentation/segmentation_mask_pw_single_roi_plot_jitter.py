import cv2
import numpy as np
import mediapipe as mp

import DEFINITION_FACEMASK
import helper_code as helper

import time
import matplotlib.pyplot as plt
from OneEuroFilter import OneEuroFilter
import copy
import scipy.signal

# from mediapipe.python._framework_bindings import timestamp
# Timestamp = timestamp.Timestamp


def butter_lowpass(cutoff, fs, order=5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def plot_filtered_coordinates(x_coords):
    # Filter requirements.
    order = 6
    global fs  # sample rate, Hz
    cutoff = 1.0  # 3.667  # desired cutoff frequency of the filter, Hz

    # Filter the data, and plot both the original and filtered signals.
    x_coords_lowpass_filtered = butter_lowpass_filter(x_coords, cutoff, fs, order)

    # define lowpass filter with 2.5 Hz cutoff frequency
    b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
    x_coords_lfilter = scipy.signal.lfilter(b, a, x_coords)
    # apply filter forward and backward using filtfilt
    x_coords_filtfilt = scipy.signal.filtfilt(b, a, x_coords)

    plt.plot(x_coords, 'k-', label='x coords')
    plt.plot(x_coords_lowpass_filtered, 'r-', linewidth=2, label='lowpass filtered x_coords')
    plt.plot(x_coords_lfilter, 'g-', linewidth=2, alpha=0.5, label='scipy lfilter x_coords')
    plt.plot(x_coords_filtfilt, 'b-', linewidth=2, alpha=0.8, label='scipy filtfilt x_coords')
    plt.xlabel('Frames')
    plt.grid()
    plt.legend()
    plt.show()


def plot_fft(data, data_filt, sample_rate = 30, title='FFT of signal'):
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)

    fft_result_filt = np.fft.fft(data_filt)
    freqs_filt = np.fft.fftfreq(len(fft_result_filt), 1 / sample_rate)

    plt.figure(figsize=(8, 6))
    plt.plot(freqs, np.abs(fft_result), 'tab:blue', label='FFT')
    plt.plot(freqs_filt, np.abs(fft_result_filt), 'tab:orange', alpha=0.5, label=('FFT of filtered signal'))
    # plt.xscale('log')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()


example_video = "../data/vids/angle_jitter_test_2.mp4"  # "jitter_test.mp4"  # 0  # "vid.avi"

if example_video == 0:
    cap = cv2.VideoCapture(example_video, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap = cv2.VideoCapture(example_video)
    cap_filter = cv2.VideoCapture(example_video)

# initialize moving average filter
window_size = 5
centroid_list = []

centroid_coords_forehead = []
centroid_coords_left_cheek = []
centroid_coords_right_cheek = []
centroid_coords_nose = []
centroid_coords_forehead_filtered = []
centroid_coords_left_cheek_filtered = []
centroid_coords_right_cheek_filtered = []
centroid_coords_nose_filtered = []
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

nose_coord_x = []
nose_coord_y = []

nose_coord_x_filtered = []
nose_coord_y_filtered = []

landmark_coords_xyz_history = [[] for _ in range(478)]
fs = 30

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while cap_filter.isOpened():
        success, frame = cap_filter.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            if example_video == 0:
                continue
            else:
                break
        frame = cv2.flip(frame, 1)

        img_h, img_w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Extract landmarks' xyz-coordinates from the detected face
                for index, landmark in enumerate(face_landmarks.landmark):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    landmark_coords_xyz_history[index].append((x, y, z))

# define lowpass filter with 2.5 Hz cutoff frequency
b, a = scipy.signal.iirfilter(20, Wn=2.9, fs=fs, btype="low", ftype="butter")

for idx in range(478):
    x_coords = [coords_xy[0] for coords_xy in landmark_coords_xyz_history[idx]]
    y_coords = [coords_xy[1] for coords_xy in landmark_coords_xyz_history[idx]]
    z_coords = [coords_xy[2] for coords_xy in landmark_coords_xyz_history[idx]]

    if len(x_coords) > 15:
        # if idx == 4:
        #     plot_filtered_coordinates(x_coords)

        # apply filter forward and backward using filtfilt
        x_coords_lowpass_filtered = scipy.signal.filtfilt(b, a, x_coords)
        y_coords_lowpass_filtered = scipy.signal.filtfilt(b, a, y_coords)
        z_coords_lowpass_filtered = scipy.signal.filtfilt(b, a, z_coords)

        landmark_coords_xyz_history[idx] = [(x_coords_lowpass_filtered[i], y_coords_lowpass_filtered[i], z_coords_lowpass_filtered[i]) for i in
                                            range(0, len(x_coords_lowpass_filtered))]

cap_filter.release()

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
            cX_forehead, cY_forehead = helper.calc_centroids(output_roi_forehead)  # mesh_points[4]
            centroid_coords_forehead.append((frame_cnt, (cX_forehead, cY_forehead)))
            cX_left_cheek, cY_left_cheek = helper.calc_centroids(output_roi_left_cheek)
            centroid_coords_left_cheek.append((frame_cnt, (cX_left_cheek, cY_left_cheek)))
            cX_right_cheek, cY_right_cheek = helper.calc_centroids(output_roi_right_cheek)
            centroid_coords_right_cheek.append((frame_cnt, (cX_right_cheek, cY_right_cheek)))

            # filtered ROI's centroids
            # ROI forehead
            output_roi_forehead_filtered = helper.segment_roi(frame, mesh_points_forehead_filtered)
            cX_forehead_filtered, cY_forehead_filtered = helper.calc_centroids(output_roi_forehead_filtered)  # mesh_points_filtered[4]
            centroid_coords_forehead_filtered.append((frame_cnt, (cX_forehead_filtered, cY_forehead_filtered)))

            # ROIs of total face
            # drawing on the mask
            cv2.fillPoly(mask_face, mesh_points_forehead, (255, 255, 255, cv2.LINE_AA))
            cv2.fillPoly(mask_face, mesh_points_left_cheek, (255, 255, 255, cv2.LINE_AA))
            cv2.fillPoly(mask_face, mesh_points_right_cheek, (255, 255, 255, cv2.LINE_AA))
            output_roi_face = cv2.copyTo(frame, mask_face)

            # drawing ROI on the frames
            '''
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
            '''

            # cv2.circle(frame, (cX_forehead, cY_forehead), 3, (0, 0, 255), -1)
            # cv2.circle(frame, (cX_forehead_filtered, cY_forehead_filtered), 1, (0, 255, 0), -1)

            for point in mesh_points:
                cv2.circle(frame, (point), 1, (0, 0, 255), -1)

            # plot filtfilt filtered landmark coordinates
            for face_landmarks in landmark_coords_xyz_history:
                # Extract landmarks' xyz-coordinates from the detected face
                cv2.circle(frame, (int(face_landmarks[frame_cnt][0]*img_w), int(face_landmarks[frame_cnt][1]*img_h)), 1, (0, 255, 0), -1)  # img_w, img_h

            # get coordinates of nose tip landmark to plot its jitter
            landmark_idx = 4
            cX_nose, cY_nose = mesh_points[landmark_idx]
            centroid_coords_nose.append((frame_cnt, (cX_nose, cY_nose)))
            cX_nose_filtered, cY_nose_filtered = int(landmark_coords_xyz_history[landmark_idx][frame_cnt][0] * img_w), \
                                                 int(landmark_coords_xyz_history[landmark_idx][frame_cnt][1] * img_h)
            centroid_coords_nose_filtered.append((frame_cnt, (cX_nose_filtered, cY_nose_filtered)))

            nose_coord_x.append(mesh_points_filtered[4][0])
            nose_coord_y.append(mesh_points_filtered[4][1])

            nose_coord_x_filtered.append(cX_nose_filtered)
            nose_coord_y_filtered.append(cY_nose_filtered)

            # used for x-axis of plot
            frame_cnt += 1

            # for point in mesh_points_filtered:
            #     cv2.circle(frame, (point), 1, (0, 255, 0), -1)

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
helper.plot_jitter_comparison(centroid_coords_nose, centroid_coords_nose_filtered, "nose tip")
# helper.plot_jitter_comparison(centroid_coords_forehead, centroid_coords_forehead_filtered, "forehead")

plt.show()

# Plot the FFT of nose landmark's x coordinate
plot_fft(nose_coord_x, nose_coord_x_filtered, sample_rate=30, title='FFT of nose landmarks x-coordinate')

# Calculate the FFT of nose landmark's y coordinate
plot_fft(nose_coord_y, nose_coord_y_filtered, sample_rate=30, title='FFT of nose landmarks y-coordinate')

