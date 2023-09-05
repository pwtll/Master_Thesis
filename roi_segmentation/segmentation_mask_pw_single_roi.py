import cv2
import numpy as np
import mediapipe as mp

import DEFINITION_FACEMASK
import helper_code as helper
import time


example_video = 0  # "vid.avi"

if example_video == 0:
    cap = cv2.VideoCapture(example_video, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap = cv2.VideoCapture(example_video)

# initialize moving average filter
window_size = 5
centroid_list = []

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
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        img_h, img_w = frame.shape[:2]
        mask_face = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_eyes = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_mouth = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_eyebrows = np.zeros((img_h, img_w), dtype=np.uint8)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            # define mesh points of each ROI
            mesh_points_forehead = [mesh_points[DEFINITION_FACEMASK.FOREHEAD_list]]
            mesh_points_left_cheek = [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.LEFT_CHEEK_LIST,
                                                                     landmarks=results.multi_face_landmarks[0].landmark,
                                                                     img_w=img_w, img_h=img_h)]
            mesh_points_right_cheek = [helper.generate_contour_points(flexible_list=DEFINITION_FACEMASK.RIGHT_CHEEK_LIST,
                                                                      landmarks=results.multi_face_landmarks[0].landmark,
                                                                      img_w=img_w, img_h=img_h)]

            # isolate single ROIs from frame
            # ROI forehead
            output_roi_forehead = helper.segment_roi(frame, mesh_points_forehead)
            # ROI left cheek
            output_roi_left_cheek = helper.segment_roi(frame, mesh_points_left_cheek)
            # ROI right cheek
            output_roi_right_cheek = helper.segment_roi(frame, mesh_points_right_cheek)

            # ROIs of total face
            # drawing on the mask
            cv2.fillPoly(mask_face, mesh_points_forehead, (255, 255, 255, cv2.LINE_AA))
            cv2.fillPoly(mask_face, mesh_points_left_cheek, (255, 255, 255, cv2.LINE_AA))
            cv2.fillPoly(mask_face, mesh_points_right_cheek, (255, 255, 255, cv2.LINE_AA))
            output_roi_face = cv2.copyTo(frame, mask_face)

            # drawing ROI on the frames
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

            # ToDo: untersuche die Auswirkung von verschiedenen Interpolationen (INTER_AREA, INTER_CUBIC, INTER_LINEAR)
            # output_roi_face = cv2.resize(output_roi_face, (36, 36))

            frame = frame[int(cY - distance_max / 2):int(cY + distance_max / 2),
                          int(cX - distance_max / 2):int(cX + distance_max / 2)]
            # frame = cv2.resize(frame, (36, 36))

            # ToDo: wirft exception, wenn Gesicht aus dem Bild geht
            try:
                # crop ROI frames

                # output_roi_face = cv2.resize(output_roi_face, (36, 36))
                output_roi_forehead = helper.resize_roi(output_roi_forehead, mesh_points_forehead)
                output_roi_left_cheek = helper.resize_roi(output_roi_left_cheek, mesh_points_left_cheek)
                output_roi_right_cheek = helper.resize_roi(output_roi_right_cheek, mesh_points_right_cheek)

                cv2.imshow('img', frame)
                cv2.imshow('ROI face', output_roi_face)
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
