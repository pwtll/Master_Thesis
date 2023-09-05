import cv2
import mediapipe as mp
import sklearn.preprocessing
import DEFINITION_FACEMASK
import helper_code as helper
import time
from surface_normal_vector import helper_functions
from rPPG_Toolbox.unsupervised_methods.methods.CHROME_DEHAAN import *
from rPPG_Toolbox.unsupervised_methods.methods.POS_WANG import *
import matplotlib.pyplot as plt


def calculate_roi(results, image, threshold=90):
    mesh_points_apaptive_roi_ = []

    for face_landmarks in results.multi_face_landmarks:

        # Extract landmarks' xyz-coordinates from the detected face
        landmark_coords_xyz = []
        for landmark in face_landmarks.landmark:
            x, y, z = landmark.x, landmark.y, landmark.z
            landmark_coords_xyz.append([x, y, z])

        # Calculate angles between camera and surface normal vectors for whole face mesh tessellation and draw an angle heatmap
        for triangle in DEFINITION_FACEMASK.FACE_MESH_TESSELATION:
            # calculate reflectance angle in degree
            angle_degrees = helper_functions.calculate_angle_heatmap(landmark_coords_xyz, triangle)

            if angle_degrees < threshold:
                # Extract the coordinates of the three landmarks of the triangle
                triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, triangle)
                mesh_points_apaptive_roi_.append(triangle_coords)

    return mesh_points_apaptive_roi_


def main(video_file=None, fps=30, threshold=90):
    if video_file == 0 or video_file is None:
        cap = cv2.VideoCapture(video_file, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(video_file)

    mp_face_mesh = mp.solutions.face_mesh

    frames = list()
    times = list()
    video_frame_count = 0

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        startTime = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                if video_file == 0:
                    continue
                else:
                    break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                video_frame_count += 1

                # define mesh points of each ROI if mesh triangles are below threshold
                mesh_points_apaptive_roi = calculate_roi(results, frame, threshold=threshold)

                # create mask of isolated ROIs below defined threshold from frame
                output_roi_face = helper.segment_roi(frame, mesh_points_apaptive_roi)

                # crop frame to square bounding box, centered at centroid between all ROIs
                x_min, y_min, x_max, y_max = helper.get_bounding_box_coordinates(output_roi_face, results)
                distance_max = max(x_max - x_min, y_max - y_min)

                bb_offset = 10
                output_roi_face = output_roi_face[int((y_min+y_max-distance_max)/2-bb_offset):int((y_min+y_max+distance_max)/2+bb_offset),
                                                  int((x_min+x_max-distance_max)/2-bb_offset):int((x_min+x_max+distance_max)/2+bb_offset)]

                # ToDo: untersuche die Auswirkung von verschiedenen Interpolationen (INTER_AREA, INTER_CUBIC, INTER_LINEAR)
                # resize ROI frames into suitable dimensions for Convolutional Attention Network
                # ts_can_dimension = 36
                # output_roi_face = cv2.resize(output_roi_face, (ts_can_dimension, ts_can_dimension))

                # frame = frame[int((y_min+y_max-distance_max)/2):int((y_min+y_max+distance_max)/2),
                #               int((x_min+x_max-distance_max)/2):int((x_min+x_max+distance_max)/2)]
                # frame = cv2.resize(frame, (36, 36))

                # ToDo: wirft exception, wenn Gesicht aus dem Bild geht
                try:
                    cv2.imshow('img', frame)
                    cv2.imshow('ROI face', output_roi_face)

                    frame = cv2.cvtColor(np.array(output_roi_face), cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    times.append(time.time() - startTime)
                    # print(1/(time.time() - startTime))
                except cv2.error:
                    pass

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    # calculate BVP with POS algorithm
    BVP = POS_WANG(frames, fs=fps)

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # BVP = BVP[fps:]  # nur zum debuggen
    BVP = scaler.fit_transform(BVP.reshape(-1, 1))

    # plot BVP
    fig, ax = plt.subplots()
    plt.plot(times, BVP)
    plt.ylabel("BVP Amplitude")
    plt.xlabel("Time [s]")
    plt.legend(['rPPG'])
    plt.show()


if __name__ == "__main__":

    # threshold angle in degrees. If the calculated angle is below this threshold, the heatmap will be drawn on the image
    threshold = 90

    fps = 20

    video_file = 0

    main(video_file=video_file, fps=fps, threshold=threshold)
