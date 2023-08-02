import cv2
import mediapipe as mp
from roi_segmentation.DEFINITION_FACEMASK import FACE_MESH_TESSELATION
import helper_functions


def main():
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract landmarks' coordinates from the detected face
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        landmarks.append([x, y, z])

                    # Calculate angles between camera and surface normal vectors for whole face mesh tessellation
                    # save angle of each mesh triangle in a dict
                    angle_dict = {}
                    for triangle in FACE_MESH_TESSELATION:
                        # draw_landmarks(image, face_landmarks, triangle)

                        # create angle heatmap
                        angle_degrees = helper_functions.draw_angle_heatmap(image, landmarks, triangle, threshold=90)
                        angle_dict.update({str(triangle): angle_degrees})

                    # get mesh triangles with lowest angles
                    angles_list = sorted(list(angle_dict.values()))
                    lowest_angles = angles_list[:10]

                    # colorize mesh triangles with lowest angles
                    hue = 255      # the lower the angle, the brighter the colorization
                    for val in lowest_angles:
                        triangle_coords = helper_functions.get_triangle_coords(image, landmarks, helper_functions.get_triangle_indices_from_angle(angle_dict, val))
                        cv2.polylines(image, [triangle_coords], True, (0, hue, 0), thickness=1)
                        hue -= 25

            cv2.imshow("Face Mesh Tesselation", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
