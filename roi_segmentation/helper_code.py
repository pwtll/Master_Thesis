import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import List, Collection, NamedTuple

import scipy.interpolate
from matplotlib import colors as mcol, pyplot as plt, cm as cm
from scipy.spatial import distance, Delaunay
from scipy.interpolate import griddata

import roi_segmentation.DEFINITION_FACEMASK
from numba import jit


@jit(nopython=True)
def triangle_centroid(triangle):
    """
    Calculate the centroid of a triangle.
    It should have three rows, each containing the (x, y) coordinates of one of its vertices.
    The centroid is computed as the average (mean) of the vertices' coordinates along each axis (x and y).

    Example:
    triangle = np.array([[0, 0], [1, 0], [0, 1]])
    centroid = triangle_centroid(triangle)
    # centroid is now [0.33333333, 0.33333333]

    :param triangle: (numpy.ndarray): A 2D numpy array representing a triangle.
    :return: numpy.ndarray: A 1D numpy array representing the centroid of the triangle.
    """
    return np.mean(triangle, axis=0)


def euclidean_distance(triangle1, triangle2):
    """
    Calculate the Euclidean distance between the centroids of two triangles.
    Each triangle should have three rows, each containing the (x, y) coordinates of one of its vertices.

    :param triangle1: (numpy.ndarray): A 2D numpy array representing the first triangle.
    :param triangle2: (numpy.ndarray): A 2D numpy array representing the second triangle.
    :return: float: The Euclidean distance between the centroids of the two triangles.
    """
    centroid1 = triangle_centroid(triangle1)
    centroid2 = triangle_centroid(triangle2)
    return distance.euclidean(centroid1, centroid2)


def apply_convex_hull(mask_roi):
    """
    Applies the convex hull algorithm to the provided masked region of interest (ROI) to smooth the contour of the masked area. It calculates the convex
    hull for the input contours and fills the convex hull with white color, resulting in a smoothed mask.

    :param mask_roi: (numpy.ndarray): The input masked region of interest.
    :return: numpy.ndarray: The ROI with the convex hull applied.
    """
    # extracted and smoothed ROI triangles with cv2.convexHulls
    contours, hierarchy = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # RETR_EXTERNAL    RETR_TREE

    # create hull array for convex hull points
    hull = []
    # calculate points for each contour
    for i in np.arange(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], clockwise=False))
    # mask_roi = mask_roi.copy()
    mask_roi = cv2.drawContours(mask_roi, hull, -1, (255, 255, 255), thickness=cv2.FILLED)

    return mask_roi


def count_pixel_area_cv(mask_image):
    """
    Counts the pixel area of a masked image.

    This function takes a masked image and counts the pixel area of the masked region.
    It first converts the image to grayscale if it's in color and then applies a threshold
    to separate black and non-black pixels. The non-black pixel count is returned as the
    pixel area of the masked region.

    :param mask_image: (numpy.ndarray): The masked image.
    :return: int: The pixel area of the masked region.
    """
    if len(mask_image.shape) == 3:
        # Convert the image to grayscale for simplicity (assuming it's a black and white image)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate black and non-black pixels (adjust threshold value as needed)
    _, thresholded_image = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)

    # Count the non-black pixels
    return cv2.countNonZero(thresholded_image)


# no jit, because opencv code is faster than numpy and numba
def count_pixel_area(mask_image):
    """
    Counts the pixel area of a masked image.

    This function takes a masked image and counts the pixel area of the masked region.
    It first converts the image to grayscale if it's in color and then applies a threshold
    to separate black and non-black pixels. The non-black pixel count is returned as the
    pixel area of the masked region.

    :param mask_image: (numpy.ndarray): The masked image.
    :return: int: The pixel area of the masked region.
    """
    # Count the non-black pixels
    return cv2.countNonZero(mask_image)


def mask_eyes_out(frame, landmark_coords_xyz):  # results):
    """
    Applies a mask to hide the eyes in a given frame based on facial landmarks provided by mediapipe.
    It uses the facial landmarks to determine the contours of the eyes and applies a mask to cover those regions. The mask is returned with the eyes hidden.


    :param frame: (numpy.ndarray): The input frame (image) on which to apply the mask.
    :param results: The results from mediapipe`s facial landmark detection model
    :return: numpy.ndarray: A white mask frame with eyes hidden (in black).
    """
    img_h, img_w = frame.shape[:2]

    # Convert facial landmarks to image coordinates
    # mesh_points_eyes = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
    #                              for p in results.multi_face_landmarks[0].landmark])
    mesh_points_eyes = np.array([np.multiply([p[0], p[1]], [img_w, img_h]).astype(int)
                                 for p in landmark_coords_xyz[roi_segmentation.DEFINITION_FACEMASK.LEFT_EYE_CONTOUR_list]])
    # Create an empty mask of the same size as the frame
    mask_eyes = np.zeros((img_h, img_w), dtype=np.uint8)
    # Fill the mask to cover the left and right eye contours
    cv2.fillPoly(mask_eyes, [mesh_points_eyes], (255, 255, 255, cv2.LINE_AA))
    # cv2.fillPoly(mask_eyes, [mesh_points_eyes[roi_segmentation.DEFINITION_FACEMASK.LEFT_EYE_CONTOUR_list]], (255, 255, 255, cv2.LINE_AA))

    mesh_points_eyes = np.array([np.multiply([p[0], p[1]], [img_w, img_h]).astype(int)
                                 for p in landmark_coords_xyz[roi_segmentation.DEFINITION_FACEMASK.RIGHT_EYE_CONTOUR_list]])
    cv2.fillPoly(mask_eyes, [mesh_points_eyes], (255, 255, 255, cv2.LINE_AA))
    # cv2.fillPoly(mask_eyes, [mesh_points_eyes[roi_segmentation.DEFINITION_FACEMASK.RIGHT_EYE_CONTOUR_list]], (255, 255, 255, cv2.LINE_AA))
    # Dilate the mask to ensure the eyes are fully covered
    kernel = np.ones((3, 3), np.uint8)
    mask_eyes = cv2.dilate(mask_eyes, kernel, iterations=1)
    # Invert the mask to make the eyes regions black (hide them)
    cv2.bitwise_not(mask_eyes, mask_eyes)

    return mask_eyes


def segment_roi(img: np.ndarray, mesh_points: List[np.ndarray], use_convex_hull: bool = False) -> np.ndarray:
    """
    Segments a Region of Interest (ROI) out of an input image according to provided set of mesh_points.
    Returns an image where all pixels outside the ROI are blackened, isolating the ROI within the output image.
    Optionally, you can choose to use the convex hull of the mesh points to define the ROI.

    :param img: (np.ndarray): An image containing the ROI represented as a numpy ndarray.
    :param mesh_points: (List[np.ndarray]): List containing ndarrays of 2d coordinates representing the mesh points of the ROI
    :param use_convex_hull:(bool, optional, default=False): A flag that determines whether to use the convex hull of the mesh points to define the ROI.
    If set to True, the convex hull will be used; otherwise, the function will use the mesh points directly.
    :return: np.ndarray: An image represented as a numpy ndarray with all pixels blackened except the ROI
    """
    img_h, img_w = img.shape[:2]
    mask_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    frame_roi = img.copy()

    for point in mesh_points:
        cv2.fillConvexPoly(mask_roi, point, (255, 255, 255, cv2.LINE_AA))

    if use_convex_hull:
        # extracted and smoothed ROI triangles with cv2.convexHulls
        contours, hierarchy = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # RETR_EXTERNAL    RETR_TREE

        # create hull array for convex hull points
        hull = []
        # calculate points for each contour
        for i in np.arange(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], clockwise=False))
        mask_roi = mask_roi.copy()
        mask_roi = cv2.drawContours(mask_roi, hull, -1, (255, 255, 255), thickness=cv2.FILLED)

    output_roi = cv2.copyTo(frame_roi, mask_roi)

    return output_roi


def resize_roi(img: np.ndarray, mesh_points: List[np.ndarray]) -> np.ndarray:
    """
    Processes an image and returns a resized image with a dimension of 36x36 pixels.
    The centroid of the ROI is oriented at the center of the resized image.

    :param img: An image containing one ROI represented as a numpy ndarray.
    :param mesh_points: List containing ndarrays of 2d coordinates of the mesh_points of the ROI
    :return: np.ndarray: resized image with a dimension 36x36
    """

    x_min = mesh_points[0][:, 0].min()
    y_min = mesh_points[0][:, 1].min()
    x_max = mesh_points[0][:, 0].max()
    y_max = mesh_points[0][:, 1].max()

    distance_max = max(x_max - x_min, y_max - y_min)

    cX, cY = calc_centroids(img)

    # crop frame to square bounding box, centered at centroid
    cropped_img = img[int(cY - distance_max / 2):int(cY + distance_max / 2), int(cX - distance_max / 2):int(cX + distance_max / 2)]

    # ToDo: untersuche die Auswirkung von verschiedenen Interpolationen (INTER_AREA, INTER_CUBIC, INTER_LINEAR)
    resized_image = cv2.resize(cropped_img, (36, 36))

    return resized_image


@jit(nopython=True)
def check_acceptance(index, angle_degrees, angle_history, threshold=90):
    """
    Check whether a triangle with a given angle is accepted based on its previous angle data.
    It calculates the mean angle and standard deviation of its previous angle values and checks if:
    1. current angle is below the threshold
    2. the triangle has appeared during the last 5 frames
    3. the mean angle of the last 5 frames is below the angle threshold + one standard deviation of the mean (last_5_mean < threshold + last_5_std_dev)

    :param index: (int): The index corresponding to the triangle in the DEFINITION_FACEMASK.FACE_MESH_TESSELATION
    :param angle_degrees: (float): The surface reflectance angle of the current triangle in degrees
    :param angle_history: (numpy.ndarray): An array storing the last 5 reflectance angles of the triangle
    :param threshold:(int, optional, default=90): The angle threshold in degrees. Triangles with angles below this threshold will be included in the adaptive ROI
    :return: bool: True if the triangle is accepted, False otherwise.
    """

    angle_history[index] = np.roll(angle_history[index], shift=-1)  # Shift the values to make room for the new angle
    angle_history[index][-1] = angle_degrees  # Store the new angle in the array
    mean_angle = np.mean(angle_history[index])
    std_dev = np.std(angle_history[index])

    # Check if there are no zero values in angle_history[index] (occurs at first initialization)
    # and count how many past angle values are less than threshold + std_dev
    # ToDo: verbessere past_appearance, sodass es die gleiche boolsche bedingung hat, wie die if-Abfrage zum Akzeptieren der Dreiecke
    if np.count_nonzero(angle_history[index] == 0) == 0:
        past_appearance = angle_history[index] < threshold + std_dev
        # or (np.count_nonzero(angle_history[index][:-1] == 0) == 0 and np.mean(angle_history[index][:-1]) < threshold + np.std(angle_history[index][:-1]))
        past_appearance_count = np.count_nonzero(past_appearance)
    else:
        past_appearance_count = 0

    # accept triangle:
    # if its angle is below threshold,
    # or if it already appeared during the last 5 frames,
    # or if its mean angle during the last 5 frames is below threshold
    if angle_degrees < threshold \
            or past_appearance_count > 0 \
            or (np.count_nonzero(angle_history[index] == 0) == 0 and mean_angle < threshold + std_dev):
        return True
    return False


def interpolate_surface_normal_angles_slow(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max):
    # Perform Delaunay triangulation on the centroid coordinates.
    tri = Delaunay(centroid_coordinates)
    # Initialize an array to store the interpolated surface normal angles for each pixel.
    interpolated_surface_normal_angles = np.zeros(len(pixel_coordinates), dtype=np.float64)
    # Iterate through each pixel coordinate for interpolation.
    for i, pixel_coord in enumerate(pixel_coordinates):
        # Find the triangle that contains the current pixel using Delaunay triangulation.
        simplex_index = tri.find_simplex(pixel_coord)

        # ToDo: bei überlappenden Pixeln wegen zu großen Kopfdrehungen nur die niedrigeren Winkel nehmen
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
    # interpolated_surface_normal_angles[interpolated_surface_normal_angles == 0] = None  # 0  # None

    return interpolated_surface_normal_angles


@jit(nopython=True)
def calc_triangle_centroid_coordinates(triangle_centroid, img_w, img_h, x_min, y_min):
    """
    Calculate the coordinates of a triangle centroid relative to a specified bounding box.

    This function takes the coordinates of a triangle centroid and adjusts them to be relative to a specified
    bounding box defined by its minimum (x_min, y_min) coordinates. The resulting coordinates are suitable for
    indexing within the bounding box.

    Parameters:
    - triangle_centroid (list): The original coordinates of the triangle centroid.
    - img_w (int): The width of the image or bounding box.
    - img_h (int): The height of the image or bounding box.
    - x_min (int): The minimum x-coordinate of the bounding box.
    - y_min (int): The minimum y-coordinate of the bounding box.

    Returns:
    - list: The adjusted coordinates of the triangle centroid relative to the bounding box.
    """
    triangle_centroid[0] = int(triangle_centroid[0] * img_w - x_min)
    triangle_centroid[1] = int(triangle_centroid[1] * img_h - y_min)
    return triangle_centroid[:2]


@jit(nopython=True)
def calc_barycentric_coords(pixel_coord, centroid_coordinates, vertices):
    """
    Calculate the barycentric coordinates of a pixel within a triangle.

    Given a pixel's (x, y) coordinates, the coordinates of a triangle's vertices, and the triangle's centroid coordinates,
    this function calculates the barycentric coordinates of the pixel within the triangle.

    Parameters:
    - pixel_coord (tuple): The (x, y) coordinates of the pixel.
    - centroid_coordinates (numpy.ndarray): The (x, y) coordinates of the triangle's centroid.
    - vertices (numpy.ndarray): The (x, y) coordinates of the triangle's vertices.

    Returns:
    - numpy.ndarray: The barycentric coordinates (alpha, beta, gamma) of the pixel within the triangle.
    """
    A = np.column_stack((centroid_coordinates[vertices], np.ones(3)))
    b = np.array([pixel_coord[0], pixel_coord[1], 1], dtype="float64")

    barycentric_coords = np.linalg.solve(A.T, b)

    return barycentric_coords


def interpolate_surface_normal_angles(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max):
    """
    Interpolate surface normal angles for all pixels in the face based on Delaunay triangulation.

    This function calculates the interpolated surface normal angles for pixels within the face, bounded by the given x_min and x_max range.
    Based on Delaunay triangulation of all triangles centroid coordinates and barycentric coordinates.

    Parameters:
    - centroid_coordinates (numpy.ndarray): Array containing the centroid coordinates of all face tesselation triangles
    - pixel_coordinates (numpy.ndarray): Array of pixel coordinates to be interpolated
    - surface_normal_angles (numpy.ndarray): Array of surface normal angles for each triangle.
    - x_min (int): The minimum x-coordinate of the face bounding box
    - x_max (int): The maximum x-coordinate of the face bounding box

    Returns:
    - numpy.ndarray: A 2D array containing the interpolated surface normal angles for each face pixel
    """
    # Perform Delaunay triangulation on the centroid coordinates.
    tri = Delaunay(centroid_coordinates)

    # Find the simplex (triangle) that contains each pixel using Delaunay.
    simplex_indices = tri.find_simplex(pixel_coordinates)

    # Find the vertices of the triangles
    triangle_vertices = tri.simplices[simplex_indices]

    # Calculate the barycentric coordinates for all pixels
    barycentric_coords = np.array([calc_barycentric_coords(pixel_coord, centroid_coordinates, vertices)
                                   for pixel_coord, vertices in zip(pixel_coordinates, triangle_vertices)])

    # Interpolate the surface normal angles for all valid pixels
    interpolated_angles = np.sum(barycentric_coords * surface_normal_angles[triangle_vertices], axis=1)

    return reshape_interpolated_angles(interpolated_angles, simplex_indices, x_max, x_min)


@jit(nopython=True)
def reshape_interpolated_angles(interpolated_angles, simplex_indices, x_max, x_min):
    """
    Reshape the interpolated surface normal angles from flattened form into a 2D image like array with the width of (x_max-x_min)
    The surface normal angles of invalid pixels outside of the face are set to zero.

    Parameters:
    - interpolated_angles (numpy.ndarray): The interpolated surface normal angles for pixels.
    - simplex_indices (numpy.ndarray): Indices of the triangles containing each pixel.
    - x_max (int): The maximum horizontal bound.
    - x_min (int): The minimum horizontal bound.

    Returns:
    - numpy.ndarray: The reshaped array of interpolated surface normal angles for valid pixels.
    """
    # Filter out pixels that are outside all triangles
    invalid_pixel_indices = simplex_indices == -1
    interpolated_angles[invalid_pixel_indices] = 0

    # Create the output array with shape (-1, x_max - x_min)
    interpolated_surface_normal_angles = np.reshape(interpolated_angles, (-1, x_max - x_min))

    return interpolated_surface_normal_angles


def interpolate_surface_normal_angles_scipy(centroid_coordinates, pixel_coordinates, surface_normal_angles, x_min, x_max):
    """
    Interpolate surface normal angles for all pixels in the face using scipy's griddata interpolation.

    This function calculates the interpolated surface normal angles for pixels within the face, bounded by the given x_min and x_max range,
    using scipy's griddata interpolation. In the end the interpolated surface normal angles are reshaped from flattened form into a 2D image like array
    with the width of (x_max-x_min). The surface normal angles of pixels outside of the face are set to zero.

    Parameters:
    - centroid_coordinates (numpy.ndarray): Array containing the centroid coordinates of all face tesselation triangles
    - pixel_coordinates (numpy.ndarray): Array of pixel coordinates to be interpolated
    - surface_normal_angles (numpy.ndarray): Array of surface normal angles for each triangle.
    - x_min (int): The minimum x-coordinate of the face bounding box
    - x_max (int): The maximum x-coordinate of the face bounding box

    Returns:
    - numpy.ndarray: A 2D array containing the interpolated surface normal angles for each face pixel
    """
    # Create a meshgrid for centroid_coordinates
    x_coords, y_coords = centroid_coordinates[:, 0], centroid_coordinates[:, 1]

    # Perform griddata interpolation to obtain the interpolated angles
    # The surface normal angles of invalid pixels outside of the face are set to zero
    interpolated_angles = griddata((x_coords, y_coords), surface_normal_angles, (pixel_coordinates[:, 0], pixel_coordinates[:, 1]), method='linear', fill_value=0)

    # Reshape the interpolated surface normal angles from flattened form into a 2D image like array with the width of (x_max-x_min)
    return np.reshape(interpolated_angles, (-1, x_max - x_min))


def plot_interpolation_heatmap(interpolated_surface_normal_angles, xx, yy):
    """
    Plot a heatmap of interpolated surface normal angles with contour lines.

    This function visualizes the interpolated surface normal angles using a heatmap and overlays contour lines at
    15-degree intervals between 0° and 90°. It also adds a colorbar to represent the angle values.

    Parameters:
    - interpolated_surface_normal_angles (numpy.ndarray): The interpolated surface normal angles for pixels.
    - xx (numpy.ndarray): X-coordinates of the pixel grid.
    - yy (numpy.ndarray): Y-coordinates of the pixel grid.

    Returns:
    - None: This function displays the plot and does not return any values.
    """
    # Make a user-defined colormap (source: https://stackoverflow.com/questions/25748183/python-making-color-bar-that-runs-from-red-to-blue)
    # cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "g", "r"])

    plt.imshow(interpolated_surface_normal_angles, cmap=cm1)  # , interpolation='nearest')    cmap='RdBu'    , cmap='seismic_r'
    create_colorbar(cm1, interpolated_surface_normal_angles)

    # plot contour lines each 30° between 0° to 90°
    CS = plt.contour(xx, yy, interpolated_surface_normal_angles, np.arange(90, step=30), colors="k", linewidths=0.75)
    plt.clabel(CS, inline=1, fontsize=12)

    # plt.show()
    plt.pause(.1)
    plt.draw()
    plt.clf()


def create_colorbar(cm1, interpolated_surface_normal_angles):
    """
    Create a colorbar representing surface normal angles.

    This function generates a colorbar to represent the surface normal angles displayed in the heatmap.
    It sets appropriate tick values and labels for the colorbar.

    Parameters:
    - cm1 (matplotlib.colors.Colormap): The colormap used for the heatmap.
    - interpolated_surface_normal_angles (numpy.ndarray): The interpolated surface normal angles for pixels.

    Returns:
    - None: This function adds a colorbar to the current plot and does not return any values.
    """
    max_angle, min_angle, v = set_colorbar_ticks(interpolated_surface_normal_angles)

    # Make a normalizer that will map the angle values from [0,90] -> [0,1]
    cnorm = mcol.Normalize(vmin=min_angle, vmax=max_angle)
    # Turn these into an object that can be used to map time values to colors and can be passed to plt.colorbar()
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    cbar = plt.colorbar(cpick, label="reflectance angle (°)", ticks=v)
    cbar.set_label(label="reflectance angle (°)", size=14)  # , weight='bold')
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=14)
    plt.axis('off')


@jit(nopython=True)
def set_colorbar_ticks(interpolated_surface_normal_angles):
    """
    Calculate and set appropriate ticks for the colorbar based on interpolated surface normal angles.

    This function calculates suitable tick values for the colorbar based on the range of interpolated surface normal angles.
    It aims to create evenly spaced ticks that cover the range while considering specific angle values between the min_angle and max_angle in 15° steps.

    Parameters:
    - interpolated_surface_normal_angles (array-like): The interpolated surface normal angles for pixels.

    Returns:
    - max_angle (float): The highest interpolated surface normal angle.
    - min_angle (float): The lowest interpolated surface normal angle.
    - v (numpy.ndarray): An array of tick values for the colorbar.
    """
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


# no jit possible
def extract_mask_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, mask_eyes, x_min, y_min):
    mask_interpolated_angles = subtract_optimal_roi_from_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, x_min, y_min)

    mask_eyes = cv2.bitwise_not(mask_eyes, mask_eyes)
    mask_interpolated_angles = np.clip(mask_interpolated_angles - mask_eyes, 0, 255)

    # set all zero values to None, to find the actual lowest angles
    mask_interpolated_angles[mask_interpolated_angles == 0] = None

    # find the indices of the smallest values with the same count of optimal_roi_area
    area = count_pixel_area(mask_optimal_roi)
    flat_mask = mask_interpolated_angles.flatten()
    indices_of_smallest = np.argpartition(flat_mask, area)[:area]

    # in the new array set the values at the found indices to their original values and then to 255 to create a binary grayscale mask
    rows, cols = np.unravel_index(indices_of_smallest, mask_interpolated_angles.shape)
    mask_outside_roi = np.zeros_like(mask_interpolated_angles, dtype=np.uint8)
    mask_outside_roi[rows, cols] = flat_mask[indices_of_smallest]
    # mask_outside_roi[np.unravel_index(indices_of_smallest, mask_interpolated_angles.shape)] = flat_mask[indices_of_smallest]
    mask_outside_roi[mask_outside_roi > 0] = 255

    return mask_outside_roi


@jit(nopython=True)
def subtract_optimal_roi_from_outside_roi(img_h, img_w, interpolated_surface_normal_angles, mask_optimal_roi, x_min, y_min):
    """
    Copy the smaller interpolated_surface_normal_angles array into a larger image sized array at the specified coordinates.
    Hereby ensure that the smaller interpolated_surface_normal_angles array gets sliced, if the face is partly outside of the image.
    At the end, subtract the optimal ROI mask from the resulting outside ROI mask to ensure there are no overlaps between both masks.

    This function takes the dimensions of an image, interpolated surface normal angles, an optimal ROI mask, and
    coordinates (x_min, y_min) to create a new mask. It ensures the new mask fits within the image boundaries
    and then subtracts the optimal ROI mask from it, effectively isolating the region outside the optimal ROI.

    Parameters:
    - img_h (int): The height of the image
    - img_w (int): The width of the image
    - interpolated_surface_normal_angles (numpy.ndarray): An array of interpolated surface normal angles
    - mask_optimal_roi (numpy.ndarray): The mask of the optimal ROI
    - x_min (int): The minimum x-coordinate for the position of the interpolated_surface_normal_angles array inside the new mask
    - y_min (int): The minimum y-coordinate for the position of the interpolated_surface_normal_angles array inside the new mask

    Returns:
    - mask_interpolated_angles (numpy.ndarray): The resulting mask with the original image dimensions after subtracting the optimal ROI.
    """
    # Create the larger mask_interpolated_angles array filled with zeros
    mask_interpolated_angles = np.zeros((img_h, img_w), dtype=interpolated_surface_normal_angles.dtype)

    # ensure that y_max and x_max are always smaller than or equal to image dimensions
    y_max = min(y_min + interpolated_surface_normal_angles.shape[0], img_h)
    x_max = min(x_min + interpolated_surface_normal_angles.shape[1], img_w)

    # ensure that y_min and x_min are always greater than or equal to 0
    y_min = max(y_max - interpolated_surface_normal_angles.shape[0], 0)
    x_min = max(x_max - interpolated_surface_normal_angles.shape[1], 0)

    # Copy the smaller interpolated_surface_normal_angles array into the larger mask_interpolated_angles array at the specified coordinates
    # slice interpolated_surface_normal_angles if face is partly outside of camera
    mask_interpolated_angles[y_min:y_max, x_min:x_max] = interpolated_surface_normal_angles[:y_max-y_min, :x_max-x_min]

    # subtract optimal_roi from outside_roi
    mask_interpolated_angles = np.clip(mask_interpolated_angles - mask_optimal_roi, 0, 255)
    # mask_interpolated_angles[np.isnan(mask_interpolated_angles)] = 0
    return mask_interpolated_angles


def get_bounding_box_coordinates(img: np.ndarray, results: NamedTuple) -> (int, int, int, int):
    """
    Processes an image and returns the minimum and maximum x and y coordinates of the bounding box of detected face.
    The bounding box is determined from the minimum and maximum occurring x and y values of all mediapipe landmarks in the face.

    :param img: An image represented as a numpy ndarray.
    :param results: A NamedTuple object with a "multi_face_landmarks" field that contains the face landmarks on each detected face.
    :return: x_min, y_min, x_max, y_max: minimum and maximum coordinates of face bounding box as integers
    """
    for face_landmarks in results.multi_face_landmarks:
        img_h, img_w = img.shape[0:2]
        x_min = img_w
        y_min = img_h
        x_max = y_max = 0
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * img_w), int(landmark.y * img_h)
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y

        # draw bounding box
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        return x_min, y_min, x_max, y_max


@jit(nopython=True)
def get_bounding_box_coordinates_mesh_points(mesh_points):
    """
    Returns the minimum and maximum x and y coordinates of the bounding box of detected face.
    The bounding box is determined from the minimum and maximum occurring x and y values of the extracted ROI mesh points.

    :param mesh_points (np.ndarray): Array of coordinates defining triangles of the extracted ROI.
    :return: x_min, y_min, x_max, y_max: minimum and maximum coordinates of face bounding box as integers
    """
    x_min, x_max = mesh_points[:, :, 0].min(), mesh_points[:, :, 0].max()
    y_min, y_max = mesh_points[:, :, 1].min(), mesh_points[:, :, 1].max()

    return x_min, y_min, x_max, y_max


# @jit(nopython=True)
def get_bounding_box_coordinates_filtered(img: np.ndarray, landmark_coords_xyz_history: np.ndarray, video_frame_count) -> (int, int, int, int):
    """
    Processes an image and returns the minimum and maximum x and y coordinates of the bounding box of detected face.
    The bounding box is determined from the minimum and maximum occurring x and y values of all mediapipe landmarks in the face.

    :param img: An image represented as a numpy ndarray.
    :param landmark_coords_xyz_history: A numpy ndarray with the dimensions (landmark_indices: 478, video_frame_number: N, xyz-coordinates: 3) that contains
                                        the face landmark's coordinates for every frame in the video.
    :return: x_min, y_min, x_max, y_max: minimum and maximum coordinates of face bounding box as integers
    """
    img_h, img_w = img.shape[0:2]

    x_min, x_max = int(landmark_coords_xyz_history[:, video_frame_count, 0].min() * img_w), int(landmark_coords_xyz_history[:, video_frame_count, 0].max() * img_w)
    y_min, y_max = int(landmark_coords_xyz_history[:, video_frame_count, 1].min() * img_h), int(landmark_coords_xyz_history[:, video_frame_count, 1].max() * img_h)

    return x_min, y_min, x_max, y_max


def apply_bounding_box(output_roi_face, bb_offset, x_min, y_min, x_max, y_max):
    """
        Apply a bounding box to a given region of interest (ROI) image while ensuring the box stays within the frame.
        The resulting ROI image is cropped to fit within the bounding box.

        Parameters:
        output_roi_face (numpy.ndarray): The input ROI image as a NumPy array
        bb_offset (int): The offset to apply to the bounding box borders
        x_max (int): The maximum x-coordinate of the bounding box
        x_min (int): The minimum x-coordinate of the bounding box
        y_max (int): The maximum y-coordinate of the bounding box
        y_min (int): The minimum y-coordinate of the bounding box

        Returns:
        tuple: A tuple containing:
            - output_roi_face (numpy.ndarray): The ROI image with the applied bounding box
            - x_max_bb (int): The updated maximum x-coordinate of the bounding box
            - x_min_bb (int): The updated minimum x-coordinate of the bounding box
            - y_max_bb (int): The updated maximum y-coordinate of the bounding box
            - y_min_bb (int): The updated minimum y-coordinate of the bounding box
    """
    distance_max = max(x_max - x_min, y_max - y_min)
    y_min_bb = (y_min + y_max - distance_max) / 2 - bb_offset
    y_max_bb = (y_min + y_max + distance_max) / 2 + bb_offset
    x_min_bb = (x_min + x_max - distance_max) / 2 - bb_offset
    x_max_bb = (x_min + x_max + distance_max) / 2 + bb_offset

    x_shift, y_shift = 0, 0

    # ensure that bounding box borders stay inside the frame
    if y_min_bb < 0:
        y_shift = abs(y_min_bb)
        y_max_bb = y_max_bb - y_min_bb
        y_min_bb = 0
    if y_max_bb > output_roi_face.shape[0]:
        y_shift = output_roi_face.shape[0] - y_max_bb
        y_min_bb = y_min_bb - (y_max_bb - output_roi_face.shape[0])
        y_max_bb = output_roi_face.shape[0]
    if x_min_bb < 0:
        x_shift = abs(x_min_bb)
        x_max_bb = x_max_bb - x_min_bb
        x_min_bb = 0
    if x_max_bb > output_roi_face.shape[1]:
        x_shift = output_roi_face.shape[1] - x_max_bb
        x_min_bb = x_min_bb - (x_max_bb - output_roi_face.shape[1])
        x_max_bb = output_roi_face.shape[1]

    if x_min_bb > output_roi_face.shape[1]:
        x_min_bb = output_roi_face.shape[1] - distance_max
        x_max_bb = output_roi_face.shape[1]
    if x_max_bb < distance_max:
        x_min_bb = 0
        x_max_bb = distance_max
    if y_min_bb > output_roi_face.shape[0]:
        y_min_bb = output_roi_face.shape[0] - distance_max
        y_max_bb = output_roi_face.shape[0]
    if y_max_bb < distance_max:
        y_min_bb = 0
        y_max_bb = distance_max

    # crop roi bounding_box out of image
    output_roi_face = output_roi_face[int(y_min_bb):int(y_max_bb), int(x_min_bb):int(x_max_bb)]

    if x_shift != 0 or y_shift != 0:
        # OpenCV image translation matrix
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

        # fill extracted ROI with black pixels if it's partly outside of image
        # (= translate cropped output_roi_face by x_shift & y_shift)
        shifted_roi_face = cv2.warpAffine(output_roi_face, M, (output_roi_face.shape[1], output_roi_face.shape[0]))

        return shifted_roi_face, x_max_bb, x_min_bb, y_max_bb, y_min_bb
    else:
        return output_roi_face, x_max_bb, x_min_bb, y_max_bb, y_min_bb


def calc_centroids(img: np.ndarray) -> (int, int):
    """
    Calculate centroid of each ROIs in given image and the centroid in between the calculated centroids
    Based on: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    :param img: mask image containing a single or multiple ROIs
    :return: int, int: 2d coordinates of centroid inside a single ROI or of the centroid in between multiple ROIs
    """
    # convert image to grayscale image
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
    centroid_list = []

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        centroid_list.append((cX, cY))

        cv2.circle(img, (cX, cY), 1, (0, 255, 0), -1)

    # calculate centroid between multiple rois of total face
    if len(centroid_list) > 1:
        x_coords = [p[0] for p in centroid_list]
        y_coords = [p[1] for p in centroid_list]
        _len = len(centroid_list)
        centroid_x = int(sum(x_coords) / _len)
        centroid_y = int(sum(y_coords) / _len)
        cv2.circle(img, (centroid_x, centroid_y), 1, (0, 255, 0), -1)

        return centroid_x, centroid_y
    else:
        return cX, cY


def calc_centroid_between_roi(img: np.ndarray) -> (int, int):
    """
    Calculate centroid of each ROIs in given image and the centroid in between the calculated centroids, weighted by the pixel area of each detected ROI contour
    Based on: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    :param img: mask image containing a single or multiple ROIs
    :return: int, int: 2d coordinates of centroid inside a single ROI or of the centroid in between multiple ROIs, weighted by their pixel area
    """
    # convert image to grayscale image
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img

    cX, cY = 0, 0
    centroid_list = []
    contour_area_list = []

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        centroid_list.append([cX, cY])
        contour_area_list.append(M["m00"]**2)

        # cv2.circle(img, (cX, cY), 1, (0, 255, 0), -1)

    # calculate centroid between multiple rois of total face
    if len(centroid_list) > 1:
        # Calculate the weighted average of centroids using areas as weights
        weighted_centroid = np.average(centroid_list, axis=0, weights=contour_area_list)
        centroid_x, centroid_y = int(weighted_centroid[0]), int(weighted_centroid[1])

        # cv2.circle(img, (centroid_x, centroid_y), 1, (127, 127, 127), -1)

        return centroid_x, centroid_y
    else:
        return cX, cY


def calc_centroid_of_largest_contour(img: np.ndarray) -> (int, int):
    """
    Calculate centroid of the contour with the largest area in the extracted ROIs in given image
    Based on: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    :param img: mask image containing a single or multiple ROIs
    :return: int, int: 2d coordinates of centroid inside a single ROI or of the centroid in between multiple ROIs
    """
    # convert image to grayscale image
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find the biggest countour by the area
    c = max(contours, key=cv2.contourArea)

    # calculate moments for the contour
    M = cv2.moments(c)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # cv2.circle(img, (cX, cY), 1, (0, 255, 0), -1)

    return cX, cY


def moving_average(list_: List[tuple], window_size: int) -> (int, int):
    """
    Apply moving average filter with provided window_size to provided list.
    The averaging is calculated among all first respectively second elements in the tuples inside the window_size.

    :param list_: list of 2D-tuples containing elements to be averaged
    :param window_size: specifies the number of elements in the list that are considered in the calculation of the averaging
    :return: int, int: Integer values of averaged elements in list
    """
    x = np.array([i[0] for i in list_])
    y = np.array([i[1] for i in list_])

    mov_avg_x = np.convolve(x, np.ones(window_size), 'valid') / window_size
    mov_avg_y = np.convolve(y, np.ones(window_size), 'valid') / window_size

    return int(mov_avg_x[-1]), int(mov_avg_y[-1])
