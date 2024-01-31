"""
This is a demo script to caompare several methods to filter the jittering of facial landmarks.
The current implementation plots the jitter and filtered signal of the nose tip landmark.

"""
import cv2
import mediapipe as mp
import numpy as np

from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy.signal


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_filtered_coordinates(x_coords):
    # Filter requirements.
    global fps
    order = 6
    fs = fps  # sample rate, Hz
    cutoff = 1.0  # 3.667  # desired cutoff frequency of the filter, Hz

    # Filter the data, and plot both the original and filtered signals.
    x_coords_lowpass_filtered = butter_lowpass_filter(x_coords, cutoff, fs, order)

    # define butterworth filtfilter with 3.5 Hz cutoff frequency
    b, a = scipy.signal.iirfilter(20, Wn=3.5, fs=fs, btype="low", ftype="butter")
    x_coords_lfilter = scipy.signal.lfilter(b, a, x_coords)
    # apply filter forward and backward using filtfilt
    x_coords_filtfilt = scipy.signal.filtfilt(b, a, x_coords)

    # define second butterworth filtfilter
    b, a = scipy.signal.iirfilter(4, Wn=3.5, fs=fs, btype="low", ftype="butter")
    # apply filter forward and backward using filtfilt
    x_coords_filtfilt_4th_order = scipy.signal.filtfilt(b, a, x_coords)

    # define third butterworth filtfilter
    b, a = scipy.signal.iirfilter(25, Wn=3.5, fs=fs, btype="low", ftype="butter")
    # apply filter forward and backward using filtfilt
    x_coords_filtfilt_40th_order = scipy.signal.filtfilt(b, a, x_coords)

    frame_count = np.arange(0, len(x_coords), 1)
    seconds = frame_count / 30.0
    seconds = seconds.reshape(len(x_coords), 1)


    # set the font to Charter
    font = {'family': 'serif', 'serif': ['Charter'], 'size': 12}
    plt.rc('font', **font)

    plt.plot(seconds, x_coords, 'C0-', linewidth=2, alpha=0.8, label='x-coordinates')
    # plt.plot(seconds, x_coords_lowpass_filtered, 'r-', alpha=0.5, linewidth=2, label='lowpass filtered x_coords')
    # plt.plot(seconds, x_coords_lfilter, 'g-', linewidth=2, alpha=0.5, label='scipy lfilter x_coords')
    plt.plot(seconds, x_coords_filtfilt, 'C1-', linewidth=2, alpha=0.8, label='x-coordinates lowpass filtered') #  (20th degree, f_cutoff=3 Hz)')
    # plt.plot(seconds, x_coords_filtfilt_4th_order, 'g-', linewidth=2, alpha=0.5, label='scipy filtfilt 4th order x_coords')
    # plt.plot(seconds, x_coords_filtfilt_40th_order, 'r-', linewidth=2, alpha=0.5, label='scipy filtfilt 40th order x_coords')
    plt.xlabel('Time (s)')
    plt.ylabel('x-coordinates of nose tip landmark (in pixels)')
    # plt.title("x-coordinates of nose tip landmark (in pixels)")
    plt.grid()
    plt.legend()



    plt.show()


def main(video_file=None, fps=30, threshold=90, use_convex_hull=True, constrain_roi=True, use_outside_roi=False):
    if video_file == 0 or video_file is None:
        cap_filter = cv2.VideoCapture(video_file, cv2.CAP_DSHOW)
        cap_filter.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_filter.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap_filter = cv2.VideoCapture(video_file)


    global video_frame_count
    fs = fps
    mp_face_mesh = mp.solutions.face_mesh

    landmark_coords_xyz_history = [[] for _ in range(478)]

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
                if video_file == 0:
                    continue
                else:
                    break
            frame = cv2.flip(frame, 1)

            cv2.imshow("video", frame)
            cv2.waitKey(1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # Extract landmarks' xyz-coordinates from the detected face
                    for index, landmark in enumerate(face_landmarks.landmark):
                        x, y, z = landmark.x * rgb_frame.shape[1], landmark.y * rgb_frame.shape[0], landmark.z
                        landmark_coords_xyz_history[index].append((x, y, z))

    for idx in range(478):
        x_coords = [coords_xy[0] for coords_xy in landmark_coords_xyz_history[idx]]
        y_coords = [coords_xy[1] for coords_xy in landmark_coords_xyz_history[idx]]
        z_coords = [coords_xy[2] for coords_xy in landmark_coords_xyz_history[idx]]

        if len(x_coords) > 15:
            if idx == 4:
                plot_filtered_coordinates(x_coords)

            # define lowpass filter with 3.5 Hz cutoff frequency
            b, a = scipy.signal.iirfilter(20, Wn=3.5, fs=fs, btype="low", ftype="butter")

            # apply filter forward and backward using filtfilt
            x_coords_lowpass_filtered = scipy.signal.filtfilt(b, a, x_coords)
            y_coords_lowpass_filtered = scipy.signal.filtfilt(b, a, y_coords)
            z_coords_lowpass_filtered = scipy.signal.filtfilt(b, a, z_coords)

            landmark_coords_xyz_history[idx] = [
                (x_coords_lowpass_filtered[i], y_coords_lowpass_filtered[i], z_coords_lowpass_filtered[i]) for i in
                range(0, len(x_coords_lowpass_filtered))]

    print("landmark coordinates filtered")

    cap_filter.release()


if __name__ == "__main__":
    video_file = "g:/Uni/_Master/Semester 9 (Master Thesis)/Datasets/VIPL-HR-V1/data/p100/v1/source2/video.avi"
    fps = 30

    main(video_file=video_file, fps=fps)
