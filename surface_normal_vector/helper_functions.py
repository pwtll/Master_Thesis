import os.path
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import numpy as np
import math
from roi_segmentation.DEFINITION_FACEMASK import FACE_MESH_TESSELATION
import pickle
import json
from skimage.transform import PiecewiseAffineTransform, warp


def calculate_surface_normal(landmarks, landmark_list):
    """
    Calculates the surface normal vector of a triangle defined by three landmarks (3D points) in space.
    The function takes two parameters:
    - landmarks (a dictionary of landmark coordinates)
    - landmark_list (a list of three landmark indices representing the vertices of the triangle)

    :param landmarks:  (dict): A dictionary where the keys are landmark indices, and the values are 3D coordinates (numpy arrays or lists)
                               representing the landmarks' positions in space.
    :param landmark_list: (list): A list containing three landmark indices that define the vertices of the triangle.
                                  These indices must correspond to keys present in the landmarks dictionary.
    :return: surface_normal: (numpy array): A 3D numpy array representing the surface normal vector of the triangle.
                                           The vector is normalized to have a unit length and points away from the triangle
    """

    # Extract the coordinates of the three landmarks for the surface
    points = [landmarks[i] for i in landmark_list]

    # Convert the points to NumPy array for easier computation
    points = np.array(points)

    # Calculate the vectors for two edges of the surface triangle
    v1 = points[0] - points[1]
    v2 = points[0] - points[2]

    # Calculate the surface normal vector as the cross product of the two edges
    surface_normal = np.cross(v1, v2)

    # Normalize the surface normal vector to get a unit vector
    surface_normal /= np.linalg.norm(surface_normal)

    return surface_normal


def calculate_angle_between_vectors(vector1, vector2):
    """
    Calculates the angle (in degrees) between two given vectors in 3D space using the dot product formula.

    :param vector1: (numpy array): A 3D numpy array representing the first vector.
    :param vector2: (numpy array): A 3D numpy array representing the second vector.
    :return: angle (float): The angle (in degrees) between the two input vectors.
    """

    # Calculate the angle between two vectors using the dot product formula
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return math.degrees(math.acos(cos_theta))


def calculate_angle_heatmap(landmarks, landmark_list):
    """
    Calculates the angle between the surface normal vector of a given triangle (defined by three landmarks) and the camera axis vector.
    The camera axis vector is assumed to be [0, 0, -1].
    The function returns the calculated reflectance angle (in degrees) between the surface normal vector and the camera axis vector.

    :param landmarks: (dict): A dictionary where the keys are landmark indices, and the values are 3D coordinates (numpy arrays or lists)
                              representing the landmarks' positions in space.
    :param landmark_list: (list): A list containing three landmark indices that define the vertices of the triangle.
                                  These indices must correspond to keys present in the landmarks dictionary.
    :return: angle_degrees: (float): The angle (in degrees) between the surface normal vector and the camera axis vector.
    """

    # Calculate the surface normal vector for the surface
    surface_normal = calculate_surface_normal(landmarks, landmark_list)

    # Calculate the camera axis vector
    camera_axis_vector = np.array([0, 0, -1])

    # Calculate the angle between the surface normal vector and the camera axis
    angle_degrees = calculate_angle_between_vectors(surface_normal, camera_axis_vector)

    if angle_degrees > 90:
        angle_degrees = 180 - angle_degrees

    return angle_degrees


def show_reflectance_angle_tesselation(image, landmarks, landmark_list, angle_degrees, threshold=90):
    """
    Visualizes a heatmap of the reflectance angle for a given triangle in a tesselation.
    If the reflectance angle is below a specified threshold, the function draws a heatmap on the provided image to visualize the angle.
    The reflectance angle of a given triangle is represented as a color-coded heatmap drawn on the provided image,
    where the color of the heatmap represents the angle's magnitude.

    :param image: (numpy array): A 3D NumPy array representing the input image where the heatmap will be drawn.
    :param landmarks: (dict): A dictionary where the keys are landmark indices, and the values are 3D coordinates (numpy arrays or lists)
                              representing the landmarks' positions in space.
    :param landmark_list: (list): A list containing three landmark indices that define the vertices of the triangle.
                                  These indices must correspond to keys present in the landmarks dictionary.
    :param angle_degrees: (float): The reflectance angle (in degrees) between the surface normal vector and the camera axis vector for the given triangle.
    :param threshold: (float, optional): The threshold angle in degrees. If the calculated angle is below this threshold,
                                        the heatmap will be drawn on the image. Default value is 90 degrees.
    :return: The function does not return any value.  If the calculated reflectance angle is below the threshold, the heatmap will be drawn on the image.
    """

    if angle_degrees < threshold:
        # Extract the coordinates of the three landmarks of the triangle
        triangle_coords = get_triangle_coords(image, landmarks, landmark_list)

        # convert angle to RGB values to draw heatmap
        angle_range = 90
        red = 255 * (1 - (angle_degrees / angle_range))
        blue = 255 * angle_degrees / angle_range

        # alternative colorization
        # r, g, b = rgb(angle_degrees)
        # cv2.drawContours(image, [triangle_coords], 0, (b, g, r), -1)

        cv2.drawContours(image, [triangle_coords], 0, (blue, 0, red), -1)


def rgb(value, minimum=0, maximum=90):
    """
    source: https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
    takes a numerical value and maps it to an RGB color representation based on a specified minimum and maximum range.
    The mapping is used to create a color gradient from blue to green to red, with blue corresponding to the minimum value, green at the midpoint,
    and red at the maximum value.

    :param value: (float): The input value that needs to be mapped to an RGB color representation.
    :param minimum: (float, optional): The minimum value of the range used for the color mapping. Default value is 0.
    :param maximum: (float, optional): The maximum value of the range used for the color mapping. Default value is 90.
    :return: r, g, b: (int): The color component of the RGB color (0 to 255).
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def draw_landmarks(image, face_landmarks, list_roi):
    """
    Draws landmarks on an input image based on the provided face landmarks and a list of regions of interest (ROI).
    It returns the modified image with landmarks drawn.

    :param image: (numpy array): A 2D or 3D NumPy array representing the input image on which the landmarks will be drawn.
    :param face_landmarks: (object): An object containing the face landmarks information, typically obtained using a facial landmark detection model or library.
    :param list_roi:(list): A list of landmark indices corresponding to the regions of interest (ROI) that need to be drawn on the image.
    :return: image: (numpy array): A 3D NumPy array representing the input image with landmarks drawn.
    """

    # Draw only the landmarks for the ROI list
    for i in list_roi:
        landmark_point = face_landmarks.landmark[i]
        x = int(landmark_point.x * image.shape[1])
        y = int(landmark_point.y * image.shape[0])
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

    return image


def get_triangle_coords(image, landmarks, landmark_list):
    """
    Extracts the 2D pixel coordinates of the three landmarks that define a triangle on an input image. The function takes the image,
    a dictionary of landmark coordinates, and a list of landmark indices that define the vertices of the triangle.

    :param image: (numpy array): A 3D NumPy array representing the input image on which the triangle is defined.
    :param landmarks: (dict): A dictionary where the keys are landmark indices, and the values are 3D coordinates (numpy arrays or lists) representing
                              the landmarks' positions in space.
    :param landmark_list: (list): A list containing three landmark indices that define the vertices of the triangle.
                                  These indices must correspond to keys present in the 'landmarks' dictionary.
    :return: triangle_coords: (numpy array): A 2D NumPy array containing the 2D pixel coordinates of the three landmarks that define the triangle. The array has shape (3, 2).
    """

    # Extract the coordinates of the three landmarks of the triangle
    points = [landmarks[i] for i in landmark_list]

    # Convert the points to NumPy array for easier computation
    img_h, img_w = image.shape[:2]
    points = np.array(points)
    points = np.delete(points, 2, 1)

    # coordinates are scaled to the size of the input image using its width and height to convert the normalized landmark coordinates to pixel coordinates
    points[:, 0] = (points[:, 0] * img_w)
    points[:, 1] = (points[:, 1] * img_h)

    triangle_coords = np.array([points[0][:2], points[1][:2], points[2][:2]], dtype=np.int32)

    return triangle_coords


def get_triangle_indices_from_angle(dict, val):
    """
    Returns the landmark indices corresponding to a triangle based on a specified angle value (val) from a given dictionary.

    :param dict: (dict): A dictionary containing information about different triangles, where the keys represent the vertices of each triangle,
                         and the values represent the angles between the surface normal vector and the camera axis.
    :param val: (any): The angle value associated with the triangle that you want to find the landmark indices for.
    :return: triangle_landmarks_int: (list): A list of three landmark indices representing the vertices of the triangle that matches the specified angle value (val) in the dictionary.
    """
    triangle_landmarks_ = [k for k, v in dict.items() if v == val][0]
    triangle_landmarks = triangle_landmarks_.strip('][').split(', ')

    triangle_landmarks_int = list(map(int, triangle_landmarks))
    return triangle_landmarks_int


def perspective_projection(x, y, z, fov, width, height):
    """
    Perform perspective projection of 3D (x, y, z) coordinates to 2D (u, v) coordinates.
    Source: ChatGPT

    :param x, y, z: (float): The 3D coordinates of the point to be projected.
    :param fov: (float): The field of view in degrees.
    :param width: (int): The width of the output image (viewport).
    :param height: (int): The height of the output image (viewport).

    :return: u, v: (float): The 2D coordinates (UV) after projection.
    """
    import math

    # Convert the field of view from degrees to radians
    fov_rad = math.radians(fov)

    # Calculate the focal length based on the field of view
    focal_length = width / (2 * math.tan(fov_rad / 2))

    # Calculate the projected coordinates
    u = (focal_length * x) / z + width / 2
    v = (focal_length * y) / z + height / 2

    return u, v


def plot_uv_transformation(image, uv_coords, keypoints):
    """
    Performs texture transformation from cartesian XY-coordinates to UV-coordinates on an input image using the skimage (scikit-image) library.
    The function displays the transformed texture on a new window.

    :param image: (numpy array): A 3D NumPy array representing the input image to be transformed.
    :param uv_coords: (list): A list of 2D coordinates (tuples) representing the UV coordinates corresponding to the keypoints.
                              These coordinates are used to define the transformation mapping.
    :param keypoints: (numpy array): A 2D NumPy array of shape (n, 2), where n is the number of keypoints. Each row represents a 2D keypoint coordinate.
                                     These keypoints are used to define the source and destination points for the texture transformation.

    :return: The function does not return any value. It displays the transformed texture on a new window using cv2.imshow.

    sources:
    https://github.com/apple2373/mediapipe-facemesh/blob/main/main.ipynb
    which is based on:
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html
    """

    H_new, W_new = 512, 512
    keypoints_uv = np.array([(W_new * x, H_new * y) for x, y in uv_coords], dtype=np.float32)

    # uv transformation by skimage
    tform = PiecewiseAffineTransform()
    tform.estimate(keypoints_uv, keypoints)
    texture = warp(image, tform, output_shape=(H_new, W_new))
    texture = (255 * texture).astype(np.uint8)

    # uv transformation by cv2  --> faster but works incorrectly
    ### transform_matrix = cv2.getAffineTransform(keypoints[0:3], keypoints_uv[0:3])   # should be applied to each mesh triangle
    # transform_matrix, _ = cv2.estimateAffinePartial2D(keypoints, keypoints_uv)
    # texture = cv2.warpAffine(image, transform_matrix, (H_new, W_new))

    cv2.imshow("UV transformation", texture)


def draw_tesselation_heatmap(tesselation_mean_angles, uv_map):
    """
    Generates a heatmap of a 3D tesselation (mediapipe mesh) using UV coordinates from mediapipe's canonical face model.
    The heatmap visualizes the mean reflectance angles between the camera axis and the surface normal vector of each triangle in the tesselation.
    The function takes tesselation mean reflectance angles and UV coordinates as input and returns the heatmap image.

    :param tesselation_mean_angles: (dict): A dictionary containing the mean reflectance angles of each triangle in the tesselation.
                                            The keys of the dictionary represent the triangle (typically a string representation of vertex indices),
                                            and the values represent tuples of the (mean reflectance angle, variance, sample_variance) of that triangle.
    :param uv_map: (list): A list of 2D coordinates (tuples) representing the UV coordinates of mediapipe's canonical face model.
                           Source of the uv_map: https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
    :return: image: (numpy array): A 3D NumPy array representing the heatmap image.
                            The heatmap visualizes the mean reflectance angles of each triangle in the tesselation.
    """

    # get uv coordinates of canonical model
    H_new, W_new = 512, 512
    keypoints_uv = np.array([(W_new * x, H_new * y) for x, y in uv_map], dtype=np.float32)
    image = np.zeros((H_new, W_new, 3), dtype=np.uint8)
    image.fill(255)

    # Extract the uv-coordinates of the three landmarks of each triangle
    for triangle in FACE_MESH_TESSELATION:
        landmark_coords_uv = []
        for landmark in triangle:
            landmark_coords_uv.append(keypoints_uv[landmark])
        triangle_coords_uv = np.array([landmark_coords_uv], dtype=np.int32)

        # get mean reflectance angle of mesh triangle
        triangle_mean_angle = tesselation_mean_angles[str(triangle)][0]

        # convert angle to RGB values to draw heatmap
        range = 90
        red = 255 * (1 - (triangle_mean_angle / range))
        blue = 255 * triangle_mean_angle / range
        cv2.drawContours(image, [triangle_coords_uv], 0, (blue, 0, red), -1)

    return image


def update_mean(existing_aggregate, new_value):
    """
    Implementation of Welford's online algorithm to calculate the mean value incrementally as new data points are collected.
    The algorithm updates the mean, the squared distance from the mean, and the count of samples seen so far.
    Welford's online algorithm is useful to calculate the mean and variance in a streaming fashion, avoiding storing all data points in memory.

    source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    :param existing_aggregate: (tuple): A tuple containing three values: count, mean, and M2
                                        - count (int): The current count of data points processed so far.
                                        - mean (float): The current mean value of the dataset.
                                        - M2 (float): The aggregated squared distance from the mean, used to calculate the variance.
    :param new_value:(float or numeric): The new data point to be incorporated into the mean calculation.

    :returns: (count, mean, M2) (tuple): A tuple containing the above mentioned updated values after incorporating the new data point:
                                        - count: (int): The updated count of data points processed so far.
                                        - mean: (float): The updated mean value of the dataset.
                                        - M2: (float): The updated aggregated squared distance from the mean.

    """

    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return count, mean, M2


def finalize(existing_aggregate):
    """
    Retrieve  the mean, variance, and sample variance from an aggregate obtained using Welford's online algorithm.
    The function calculates the variance and sample variance from the accumulated count, mean, and M2 values.

    source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    :param existing_aggregate: (tuple): A tuple containing three values: count, mean, and M2.
                                        These values represent the aggregate obtained using Welford's online algorithm to calculate the mean and
                                        variance incrementally.
    :return: mean, variance, sample_variance:  (tuple or float): A tuple containing the mean, variance, and sample variance calculated from the existing_aggregate.
                                                                 If the count is less than 2, which indicates insufficient data points to compute variance,
                                                                 the function returns float("nan").
    """
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        return mean, variance, sample_variance


def pickle_dump_tesselation_angles(tesselation_mean_angles, filepath='tesselation_angle_metrics_dict.pkl'):
    """
    Saves a dictionary containing tesselation mean angles to a pickle file.
    If a file with the specified filename already exists, the function appends a timestamp to the filename to avoid overwriting the existing file.

    :param tesselation_mean_angles: (dict): A dictionary containing the tesselation mean angles. The keys of the dictionary represent triangles
                                            (typically a string representation of vertex indices), and the values represent the mean angle of each triangle.
    :param filepath: (str, optional): The filepath (including the filename with extension) for the pickle file. If not provided, the default filename is 'saved_dictionary.pkl'.
    :return: The function does not return any value. It saves the tesselation_mean_angles dictionary to a pickle file in the specified filepath.
    """
    # dump file in specified directory
    with open(filepath, 'wb') as f:
        pickle.dump(tesselation_mean_angles, f)


def plot_mean_angle_heatmap_uv(tesselation_angle_metrics, uv_map, show_heatmap=True):
    """
    Visualizes a heatmap of the mean angles for each triangle in mediapipe's face tesselation using UV coordinates from the canonical face model.
    The function generates a heatmap using the draw_tesselation_heatmap function, converts it to an RGB image, and plots it.
    The function also creates a user-defined colormap for the heatmap and adds a colorbar to represent the mean angles in degrees.

    :param tesselation_mean_angles: (dict): A dictionary containing tuples of the (mean angle, variance, sample_variance) for each triangle in the tesselation.
                                            The keys of the dictionary represent the triangle (typically a string representation of vertex indices),
                                            and the values represent tuples of the (mean angle, variance, sample_variance) of each triangle in degrees.
    :param uv_map: (list): A list of 2D coordinates (tuples) representing the UV coordinates of the canonical face model.
    :return: The function does not return any value. It generates and displays the mean angle heatmap along with the colorbar.
    """
    mean_angle_heatmap_bgr = draw_tesselation_heatmap(tesselation_angle_metrics, uv_map)
    mean_angle_heatmap_rgb = cv2.cvtColor(mean_angle_heatmap_bgr, cv2.COLOR_BGR2RGB)

    plt.rcParams["figure.figsize"] = (5, 4)
    plt.subplots_adjust(left=0, bottom=0.05, right=0.95, top=0.975, wspace=0.2, hspace=0.2)
    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.serif": "Charter"})
    imgplot = plt.imshow(mean_angle_heatmap_rgb)

    '''
    # transform rgb values to angles
    for x in range(mean_angle_heatmap_rgb.shape[1]):
        for y in range(mean_angle_heatmap_rgb.shape[0]):
            # for the given pixel at w,h, lets check its value against the threshold
            red = mean_angle_heatmap_rgb[x][y][0]
            angle = 90/255*red
            mean_angle_heatmap_rgb[x][y] = angle
    imgplot = plt.imshow(mean_angle_heatmap_rgb[:,:,0])
    # plt.gca().format_coord = format_coord
    '''

    # Make a user-defined colormap (source: https://stackoverflow.com/questions/25748183/python-making-color-bar-that-runs-from-red-to-blue)
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])

    # get highest and lowest mean reflectance angles
    angles_list = sorted([tuples[0] for tuples in list(tesselation_angle_metrics.values())])
    min_angle = angles_list[0]
    max_angle = angles_list[-1]

    # Make a normalizer that will map the angle values from [0,90] -> [0,1]
    cnorm = mcol.Normalize(vmin=min_angle, vmax=max_angle)

    # Turn these into an object that can be used to map time values to colors and can be passed to plt.colorbar()
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    # set ticks for colorbar
    if min_angle < 15 - 5 and max_angle > 75 + 5:
        v = np.array([min_angle, 15, 30, 45, 60, 75, max_angle])
    elif min_angle < 15 - 5 and not max_angle > 75 + 5:
        v = np.array([min_angle, 15, 30, 45, 60, max_angle])
    elif not min_angle < 15 - 5 and max_angle > 75 + 5:
        v = np.array([min_angle, 30, 45, 60, 75, max_angle])
    elif not min_angle < 15 - 5 and not max_angle > 75 + 5:
        v = np.array([min_angle, 30, 45, 60, max_angle])

    # alternative set of ticks
    # lin_start = round(min_angle+5, -1)
    # lin_stop = 10 * math.floor(max_angle / 10)
    # v = np.linspace(lin_start, lin_stop, int((lin_stop-lin_start)/10), endpoint=True)
    # v = np.append(min_angle, v)
    # v = np.append(v, max_angle)

    plt.colorbar(cpick, label="Mean angle (°)", ticks=v)
    plt.axis('off')

    if show_heatmap:
        plt.show()

    return imgplot


'''
def format_coord(x, y, z):
    return "text_string_made_from({:.2f},{:.2f},{:.2f})".format(x, y, z)
'''


def get_video_paths_in_folder(dir):
    """
    Scans a directory and its subdirectories to find all video files with the ".avi" or ".mp4" file extension.
    It returns a list containing the full paths of all the discovered video files.

    :param dir: (str): The input string representing the directory path to be scanned for video files.
    :return: video_paths: (list): A list of full paths of video files (with the ".avi" or ".mp4" extension) found in the specified directory and its subdirectories.
    """

    video_paths = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(".avi") or name.endswith(".mp4"):
                video_paths.append(os.path.join(root, name))
                # print(os.path.join(root, name))
    return video_paths


def extract_output_folder(input_file_string):
    """
    Takes an input file path string of a dataset and extracts the subfolder structure of the path.
    The function is specifically designed to handle file paths that contain a "Datasets" directory.
    It normalizes the input path to handle different separators (e.g., '/' or '\'),
    and extracts and preserves the dataset subfolder structure as the output folder path.
    An additional subfolder with the video name is introduced to separate the saved files if there are multiple videos in the same folder.

    :param input_file_string: (str): The input string representing the file path.
    :return: output_string: (str): The output string representing the subfolder structure of the "Datasets" directory.
    """

    # Normalize the path to handle different separators (e.g., '/' or '\')
    normalized_path = os.path.normpath(input_file_string)

    # Split the path into directory components
    path_components = normalized_path.split(os.path.sep)

    # Find the index of the "Datasets" directory
    datasets_index = path_components.index("Datasets")

    # Extract the dataset structure of the path
    output_path_components = path_components[datasets_index + 1:-1]

    # Extract the video filename
    video_name = path_components[-1].split('.')[0]

    # Join the components back to form the output string
    output_string = os.path.join(*output_path_components) + "/" + video_name

    return output_string


def get_destination_path(video_file):
    """
    Generates the destination path derived from the input video file path.
    The function takes a video file path, extracts the subfolder structure of its "Datasets" directory and appends the specified filename.
    If the file already exists, the function appends a folder named with the current date-timestamp to the filepath,
    ensuring that the new files do not overwrite the existing ones.

    :param video_file: (str): The input string representing the file path of the video.
    :return: filepath: (str): The full destination path where the file with the specified filename should be saved.
    """

    directory = "../data/dataset_tesselation_angles/"
    dataset_structure = extract_output_folder(video_file)
    filename = 'tesselation_angle_metrics_dict.pkl'

    destination_folder = directory + dataset_structure + "/"
    # destination_folder = os.path.join(*destination_file.split("/")[:-2])

    # Check whether the specified path exists or not
    if not os.path.exists(destination_folder):
        # Create a new directory
        os.makedirs(destination_folder, exist_ok=True)
        # print("Created: " + destination_folder)
    else:
        # Create a new directory with the current timestamp to keep existing files
        destination_folder = directory + time.strftime("%Y%m%d") + '_' + dataset_structure + "/"
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)

    filepath = destination_folder + filename

    return filepath
