import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from numba import jit
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.colors as mcol
import matplotlib.cm as cm

from roi_segmentation.DEFINITION_FACEMASK import FACE_MESH_TESSELATION
import helper_functions


@jit(nopython=True)
def calc_triangle_centroid_coordinates(index, triangle_centroid, centroid_coordinates, img_w, img_h, x_min, y_min):
    triangle_centroid[0] = int(triangle_centroid[0] * img_w - x_min)
    triangle_centroid[1] = int(triangle_centroid[1] * img_h - y_min)
    centroid_coordinates[index] = triangle_centroid[:2]


@jit(nopython=True)
def calc_barycentric_coords(pixel_coord, centroid_coordinates, vertices):
    A = np.column_stack((centroid_coordinates[vertices], np.ones(3)))
    A.astype("float64")
    b = np.array([pixel_coord[0], pixel_coord[1], 1], dtype="float64")

    barycentric_coords = np.linalg.solve(A.T, b)

    return barycentric_coords


def plot_interpolation_heatmap(interpolated_surface_normal_angles, xx, yy):
    # Make a user-defined colormap (source: https://stackoverflow.com/questions/25748183/python-making-color-bar-that-runs-from-red-to-blue)
    # cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "g", "r"])

    plt.imshow(interpolated_surface_normal_angles, cmap=cm1)  # , interpolation='nearest')    cmap='RdBu'    , cmap='seismic_r'
    create_colorbar(cm1, interpolated_surface_normal_angles)

    # plot contour lines each 15° between 0° to 90°
    CS = plt.contour(xx, yy, interpolated_surface_normal_angles, np.arange(90, step=30), colors="k", linewidths=0.75)
    plt.clabel(CS, inline=1, fontsize=10)

    # plt.show()
    plt.pause(.1)
    plt.draw()
    plt.clf()


def create_colorbar(cm1, interpolated_surface_normal_angles):
    max_angle, min_angle, v = set_colorbar_ticks(interpolated_surface_normal_angles)

    # Make a normalizer that will map the angle values from [0,90] -> [0,1]
    cnorm = mcol.Normalize(vmin=min_angle, vmax=max_angle)
    # Turn these into an object that can be used to map time values to colors and can be passed to plt.colorbar()
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    plt.colorbar(cpick, label="Surface normal angle (°)", ticks=v)
    plt.axis('off')


@jit(nopython=True)
def set_colorbar_ticks(interpolated_surface_normal_angles):
    # get highest and lowest interpolated reflectance angles
    min_angle = np.nanmin(interpolated_surface_normal_angles)
    max_angle = np.nanmax(interpolated_surface_normal_angles)
    # alternative set of ticks
    # lin_start = round(min_angle+5, -1)
    # lin_stop = 10 * math.floor(max_angle / 10)
    # v = np.linspace(lin_start, lin_stop, int((lin_stop-lin_start)/10), endpoint=True)
    # v = np.append(min_angle, v)
    # v = np.append(v, max_angle)
    # set ticks for colorbar
    if min_angle < 15 - 5 and max_angle > 75 + 5:
        v = np.array([min_angle, 15, 30, 45, 60, 75, max_angle])
    elif min_angle < 15 - 5 and not max_angle > 75 + 5:
        v = np.array([min_angle, 15, 30, 45, 60, max_angle])
    elif not min_angle < 15 - 5 and max_angle > 75 + 5:
        v = np.array([min_angle, 30, 45, 60, 75, max_angle])
    elif not min_angle < 15 - 5 and not max_angle > 75 + 5:
        v = np.array([min_angle, 30, 45, 60, max_angle])
    return max_angle, min_angle, v


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
            img_h, img_w = image.shape[:2]

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract landmarks' coordinates from the detected face
                    landmarks = np.zeros((len(face_landmarks.landmark), 3))
                    for index, landmark in enumerate(face_landmarks.landmark):
                        landmarks[index] = [landmark.x, landmark.y, landmark.z]

                    landmarks = np.array(landmarks)
                    x_min, x_max = int(landmarks[:, 0].min() * img_w), int(landmarks[:, 0].max() * img_w)
                    y_min, y_max = int(landmarks[:, 1].min() * img_h), int(landmarks[:, 1].max() * img_h)
                    xx, yy = np.meshgrid(np.arange(x_max-x_min), np.arange(y_max-y_min))

                    # image pixel coordinates for which to interpolate surface normal angles, each row is [x, y], starting from [0, 0] -> [img_w, img_h]
                    pixel_coordinates = np.column_stack((xx.ravel(), yy.ravel()))

                    # [x, y] coordinates of the triangle centroids
                    centroid_coordinates = np.zeros((len(FACE_MESH_TESSELATION), 2), dtype=np.int32)
                    # surface normal angles for each triangle centroid
                    surface_normal_angles = np.zeros(len(FACE_MESH_TESSELATION))


                    for index, triangle in enumerate(FACE_MESH_TESSELATION):
                        # calculate reflectance angle in degree
                        angle_degrees = helper_functions.calculate_surface_normal_angle(landmarks, triangle)
                        # draw_landmarks(image, face_landmarks, triangle)

                        # display angle heatmap
                        helper_functions.show_reflectance_angle_tesselation(image, landmarks, triangle, angle_degrees, threshold=90)

                        surface_normal_angles[index] = angle_degrees
                        triangle_centroid = np.mean(np.array([landmarks[i] for i in triangle]), axis=0)
                        calc_triangle_centroid_coordinates(index, triangle_centroid, centroid_coordinates, img_w, img_h, x_min, y_min)

                        # triangle_centroid = np.mean(np.array([landmarks[i] for i in triangle]), axis=0)
                        # triangle_centroid[0] = int(triangle_centroid[0] * img_w - x_min)
                        # triangle_centroid[1] = int(triangle_centroid[1] * img_h - y_min)
                        # centroid_coordinates[index] = triangle_centroid[:2]
                    centroid_coordinates = np.array(centroid_coordinates)

                    # Perform Delaunay triangulation on the centroid coordinates.
                    tri = Delaunay(centroid_coordinates)
                    # Initialize an array to store the interpolated surface normal angles for each pixel.
                    interpolated_surface_normal_angles = np.zeros(len(pixel_coordinates), dtype=np.float64)

                    # Iterate through each pixel coordinate for interpolation.
                    for i, pixel_coord in enumerate(pixel_coordinates):
                        # Find the triangle that contains the current pixel using Delaunay triangulation.
                        simplex_index = tri.find_simplex(pixel_coord)

                        if simplex_index != -1:
                            # Get the vertices of the triangle that contains the pixel.
                            vertices = tri.simplices[simplex_index]

                            # Calculate the barycentric coordinates of the pixel within the triangle.
                            barycentric_coords = calc_barycentric_coords(pixel_coord, centroid_coordinates, vertices)

                            # Use the barycentric coordinates to interpolate the surface normal angle for the current pixel.
                            interpolated_angle = sum(barycentric_coords[i] * surface_normal_angles[vertices[i]] for i in range(3))

                            interpolated_surface_normal_angles[i] = interpolated_angle

                interpolated_surface_normal_angles = np.reshape(interpolated_surface_normal_angles, (-1, x_max - x_min))
                # interpolated_surface_normal_angles[interpolated_surface_normal_angles > 45] = 0
                interpolated_surface_normal_angles[interpolated_surface_normal_angles == 0] = None

                plot_interpolation_heatmap(interpolated_surface_normal_angles, xx, yy)
                # plot centroid triangulation
                # Xi, Yi = centroid_coordinates[:, 0], centroid_coordinates[:, 1]
                # plt.triplot(Xi, Yi, tri.simplices.copy())
                # plt.plot(Xi, Yi, "or", label="Tesselation triangle centroids", markersize=2)

            cv2.imshow("Face Mesh Tesselation", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
