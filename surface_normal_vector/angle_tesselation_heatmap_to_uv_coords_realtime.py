import cv2
import mediapipe as mp
import numpy as np
from roi_segmentation.DEFINITION_FACEMASK import FACE_MESH_TESSELATION
import json
import helper_functions


def main():
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    uv_path = "uv_map.json"  # taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
    uv_map_dict = json.load(open(uv_path))
    uv_map = np.array([(uv_map_dict["u"][str(i)], uv_map_dict["v"][str(i)]) for i in range(468)])

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.flip(image, 1)
            img_h, img_w = image.shape[:2]

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # Extract landmarks' coordinates from the detected face
                    landmark_coords_xyz = []      # xyz-coordinates
                    landmark_coords_uv = []   # uv-coordinates
                    for landmark in face_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        landmark_coords_xyz.append([x, y, z])

                        # landmark_coords_uv.append(perspective_projection(x, y, z, fov=70, width=img_w, height=img_h))


                    # Calculate angles between camera and surface normal vectors for whole face mesh tessellation
                    # save angle of each mesh triangle in a dict
                    angle_dict = {}
                    for triangle in FACE_MESH_TESSELATION:
                        # draw_landmarks(image, face_landmarks, triangle)

                        # calculate reflectance angle in degree
                        angle_degrees = helper_functions.calculate_angle_heatmap(landmark_coords_xyz, triangle)
                        angle_dict.update({str(triangle): angle_degrees})

                        # # display angle heatmap
                        helper_functions.show_reflectance_angle_tesselation(image, landmark_coords_xyz, triangle, angle_degrees, threshold=90)

                    # get mesh triangles with lowest angles
                    angles_list = sorted(list(angle_dict.values()))
                    lowest_angles = angles_list[:10]

                    # colorize mesh triangles with lowest angles
                    hue = 255      # the lower the angle, the brighter the colorization
                    for val in lowest_angles:
                        triangle_coords = helper_functions.get_triangle_coords(image, landmark_coords_xyz, helper_functions.get_triangle_indices_from_angle(angle_dict, val))
                        cv2.polylines(image, [triangle_coords], True, (0, hue, 0), thickness=1)
                        hue -= 25

                    keypoints = np.array([(img_w * point.x, img_h * point.y) for point in face_landmarks.landmark[0:468]], dtype=np.float32)  # after 468 is iris or something else
                    helper_functions.plot_uv_transformation(image, uv_map, keypoints)

            cv2.imshow("Face Mesh Tesselation", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
