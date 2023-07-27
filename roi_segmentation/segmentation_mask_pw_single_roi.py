import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
import DEFINITION_FACEMASK
import helper_code as helper

example_video = 0  # "vid.avi"

cap = cv.VideoCapture(example_video, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            if example_video == 0:
                continue
            else:
                break
        frame = cv.flip(frame, 1)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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
            cv.fillPoly(mask_face, mesh_points_forehead, (255, 255, 255, cv.LINE_AA))
            cv.fillPoly(mask_face, mesh_points_left_cheek, (255, 255, 255, cv.LINE_AA))
            cv.fillPoly(mask_face, mesh_points_right_cheek, (255, 255, 255, cv.LINE_AA))
            output_roi_face = cv.copyTo(frame, mask_face)

            # drawing ROI on the frames
            cv.polylines(frame, mesh_points_forehead, True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, mesh_points_left_cheek, True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, mesh_points_right_cheek, True, (0, 255, 0), 1, cv.LINE_AA)

            # crop frame to square bounding box, centered at centroid between all ROIs
            x_min, y_min, x_max, y_max = helper.get_bounding_box(output_roi_face, results)
            distance_max = max(x_max - x_min, y_max - y_min)
            cX, cY = helper.calc_multiple_centroids(output_roi_face)
            # crop frame to square bounding box
            output_roi_face = output_roi_face[int(cY - distance_max / 2):int(cY + distance_max / 2),
                          int(cX - distance_max / 2):int(cX + distance_max / 2)]

            # ToDo: untersuche die Auswirkung von verschiedenen Interpolationen (INTER_AREA, INTER_CUBIC, INTER_LINEAR)
            output_roi_face = cv.resize(output_roi_face, (36, 36))

            # draw bounding box
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

            # crop ROI frames
            output_roi_forehead = helper.get_roi_bounding_box(output_roi_forehead, mesh_points_forehead)
            output_roi_left_cheek = helper.get_roi_bounding_box(output_roi_left_cheek, mesh_points_left_cheek)
            output_roi_right_cheek = helper.get_roi_bounding_box(output_roi_right_cheek, mesh_points_right_cheek)

            try:
                cv.imshow('img', frame)
                cv.imshow('ROI face', output_roi_face)
                cv.imshow('ROI forehead', output_roi_forehead)
                cv.imshow('ROI left cheek', output_roi_left_cheek)
                cv.imshow('ROI right cheek', output_roi_right_cheek)
            except:
                pass

        key = cv.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
