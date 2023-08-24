import cv2
import mediapipe as mp
import numpy as np
import math

# Indices of landmarks for the surface
FOREHEAD_LIST = [107, 151, 336]
LEFT_CHEEK_LIST = [36, 111, 187]
RIGHT_CHEEK_LIST = [411, 266, 340]

def calculate_surface_normal(landmarks, landmark_list):
    # Extract the coordinates of the three landmarks for the surface
    points = [landmarks[i] for i in landmark_list]

    # Convert the points to NumPy array for easier computation
    points = np.array(points)

    # Calculate the vectors for two edges of the surface triangle
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]

    # Calculate the surface normal vector as the cross product of the two edges
    surface_normal = np.cross(v1, v2)

    # Normalize the surface normal vector to get a unit vector
    surface_normal /= np.linalg.norm(surface_normal)

    return -surface_normal

# currently unused
def calculate_camera_axis_vector(landmarks, focal_length=3):
    # Assume the camera is looking straight ahead (the focal length will be used to scale the vector)
    # The camera axis vector points from the camera position to the centroid of the forehead region
    centroid = np.mean(np.array([landmarks[i] for i in FOREHEAD_LIST]), axis=0)
    return np.array([focal_length * centroid[0], focal_length * centroid[1], focal_length])


def calculate_angle_between_vectors(vector1, vector2):
    # Calculate the angle between two vectors using the dot product formula
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return math.degrees(math.acos(cos_theta))


def calculate_and_draw_normal(image, face_landmarks, landmarks, landmark_list):
    # Calculate the surface normal vector for the surface
    surface_normal = calculate_surface_normal(landmarks, landmark_list)

    # Calculate the camera axis vector
    # camera_axis_vector = calculate_camera_axis_vector(landmarks, focal_length)
    camera_axis_vector = np.array([0, 0, -1])

    # Calculate the angle between the surface normal vector and the camera axis
    angle_degrees = calculate_angle_between_vectors(surface_normal, camera_axis_vector)

    # ToDo: richtige Vorzeichen implementieren, sodass Winkel zwischen -90° und +90° liegen
    if angle_degrees > 90:
        surface_normal = -surface_normal
        angle_degrees = 180 - angle_degrees


    # Find the centroid of the region in absolute coordinates
    centroid_x = int(np.mean([face_landmarks.landmark[i].x for i in landmark_list]) * image.shape[1])
    centroid_y = int(np.mean([face_landmarks.landmark[i].y for i in landmark_list]) * image.shape[0])

    # Scale the surface normal for visualization purposes
    scale_factor = 0.2*angle_degrees/90
    endpoint = [centroid_x + int(scale_factor * image.shape[1] * surface_normal[0]),
                centroid_y + int(scale_factor * image.shape[0] * surface_normal[1])]



    # Draw the surface normal vector on the image
    cv2.arrowedLine(image, (centroid_x, centroid_y), tuple(endpoint), (0, 255, 0), 2)

    # Display the angle between surface normal and camera axis as text
    text_angle = "{:.2f} deg".format(angle_degrees)
    cv2.putText(image, text_angle, (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def draw_landmarks(image, face_landmarks, list_roi):
    # Draw only the landmarks for the ROI list
    for i in list_roi:
        landmark_point = face_landmarks.landmark[i]
        x = int(landmark_point.x * image.shape[1])
        y = int(landmark_point.y * image.shape[0])
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # currently unused
    focal_length = 3  # Adjust this value based on your camera setup and scene

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
                    # Draw only the three landmarks for the forehead
                    draw_landmarks(image, face_landmarks, FOREHEAD_LIST)
                    draw_landmarks(image, face_landmarks, LEFT_CHEEK_LIST)
                    draw_landmarks(image, face_landmarks, RIGHT_CHEEK_LIST)

                    # Extract landmarks' coordinates from the detected face
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        landmarks.append([x, y, z])

                    # ToDo: über alle mesh_triangles iterieren
                    # Calculate and draw surface normal vectors for all three surfaces
                    calculate_and_draw_normal(image, face_landmarks, landmarks, FOREHEAD_LIST)
                    # ToDo: korrigiere Vorzeichen / ziehe Betrag bei der Winkelberechnung für alle ROIs. Der Winkel soll immer zwischen -90 bis 90° liegen° liegen
                    calculate_and_draw_normal(image, face_landmarks, landmarks, LEFT_CHEEK_LIST)
                    calculate_and_draw_normal(image, face_landmarks, landmarks, RIGHT_CHEEK_LIST)


            cv2.imshow("Face Mesh", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
