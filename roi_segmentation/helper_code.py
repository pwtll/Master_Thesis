import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

from typing import List, Dict, Any, Collection


def segment_roi(frame, mesh_points):
    img_h, img_w = frame.shape[:2]
    mask_face = np.zeros((img_h, img_w), dtype=np.uint8)

    frame_roi = frame.copy()
    mask_roi = mask_face.copy()
    cv2.fillPoly(mask_roi, mesh_points, (255, 255, 255, cv2.LINE_AA))
    output_roi = cv2.copyTo(frame_roi, mask_roi)

    calc_multiple_centroids(output_roi)

    return output_roi


def get_bounding_box(img, results):
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

        # crop frame to bounding box
        # cropped_img = img[y_min-10:y_max+10, x_min-10:x_max+10]

        return x_min, y_min, x_max, y_max


def get_roi_bounding_box(img, mesh_points, margin=0):
    x_min = mesh_points[0][:,0].min()
    y_min = mesh_points[0][:,1].min()
    x_max = mesh_points[0][:,0].max()
    y_max = mesh_points[0][:,1].max()

    distance_max = max(x_max-x_min, y_max-y_min)

    cX, cY = calc_centroids(img)

    # crop frame to square bounding box, centered at centroid
    cropped_img = img[int(cY - distance_max/2 - margin):int(cY + distance_max/2 + margin), int(cX - distance_max/2 - margin):int(cX + distance_max/2 + margin)]

    resized_image = cv2.resize(cropped_img, (36, 36))

    return resized_image


def calc_centroids(img):
    """
    source: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    :param img:
    :return:
    """
    # convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate moments of binary image
    M = cv2.moments(gray_image)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return cX, cY


def calc_multiple_centroids(img):
    """
    source: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    :param img:
    :return:
    """
    # convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    x_coords = [p[0] for p in centroid_list]
    y_coords = [p[1] for p in centroid_list]
    _len = len(centroid_list)
    centroid_x = int(sum(x_coords) / _len)
    centroid_y = int(sum(y_coords) / _len)
    cv2.circle(img, (centroid_x, centroid_y), 1, (0, 255, 0), -1)

    return centroid_x, centroid_y


def generate_contour_points(flexible_list: List, landmarks: Collection, img_w: int, img_h: int) -> np.ndarray:
    """Generates Contour Points in image coordinates based on a list of mediapipe landmark IDs, and parameters.
    The list of landmarks might also contain multiple landmarks describing a single point.
    For more information see implementation.

    Args:
        flexible_list (List): A list of: mediapipe landmark IDs defining a contour OR tuples of mediapipe landmarks and a weight OR ...
        landmarks (Collection): A Collection of landmarks as provided by mediapipe. Must contain above IDs.
        img_w (int): width of the image processed by mp
        img_h (int): height of the image processed by mp

    Returns:
        np.ndarray: Array of 2d image coordinates describing a polygonal contour
    """
    contour_points = list()
    for item in flexible_list:
        if type(item) == int:
            p = landmarks[item]
            contour_points.append(np.multiply([p.x, p.y], [img_w, img_h]).astype(int))
        elif type(item) == tuple and len(item) == 3:  # type=tuple and len=3
            p1 = np.array([landmarks[item[0]].x, landmarks[item[0]].y])
            p2 = np.array([landmarks[item[1]].x, landmarks[item[1]].y])
            alpha = item[2]
            p = np.multiply(p1 * alpha + p2 * (1 - alpha), [img_w, img_h]).astype(int)
            contour_points.append(p)
        else:
            p1 = np.array([landmarks[item[0]].x, landmarks[item[0]].y])
            p2 = np.array([landmarks[item[1]].x, landmarks[item[1]].y])
            alpha = item[2]
            diff = alpha * (p2 - p1)
            p0 = p1 + item[3] * diff
            p0[p0 > 1] = 1
            p0[p0 < 0] = 0
            p = np.multiply(p0, [img_w, img_h]).astype(int)
            contour_points.append(p)

    return np.array(contour_points)


def mask_poly_contour(contour_points: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Creates a binary Mask image based on polygon contour points (edges) in image coordinates
       and the mask height and width

    Args:
        contour_points (np.ndarray): Edge points of contour
        img_h (int): image height
        img_w (int): image width

    Returns:
        np.ndarray: binary mask
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_points], (255, 255, 255, cv2.LINE_AA))
    return mask


def get_contour_image(rgb_frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """Computes contours of masks and plots contours on rgb image

    Args:
        rgb_frame (np.ndarray): WxH RGB image
        masks (List[np.ndarray]): N WxH binary masks

    Returns:
        np.ndarray: WxH RGB image
    """
    rgb_frame_copy = rgb_frame.copy()
    for _i, _mask in enumerate(masks):
        contours, hierarchy = cv2.findContours(image=_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        if contours is not None and len(contours) > 0 and len(contours[0].shape) >= 3:
            _main_contour = contours[0][:, 0, :]
            center = np.mean(_main_contour, axis=0)
            cv2.drawContours(image=rgb_frame_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(rgb_frame_copy, str(_i + 1), center.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3);
    return rgb_frame_copy


def plot_contours(rgb_frame: np.ndarray, masks: List[np.ndarray]):
    rgb_frame_copy = get_contour_image(rgb_frame, masks)

    # see the results
    plt.figure()
    plt.imshow(rgb_frame_copy)
    plt.show()


# up to 15 overlapping masks
def stack_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Produces an image containing all masks layers in bytes.
    Max number of masks is 15 because 16 bit image is used

    Args:
        masks (List[np.ndarray]): N  WxH binary masks

    Returns:
        np.ndarray: WxH 16 bit image with N embedded masks
    """
    assert len(masks) < 16
    multimask_image = np.zeros(masks[0].shape, dtype=np.uint16)
    base = np.ones_like(masks[0])
    for i, mask in enumerate(masks):
        multimask_image = np.bitwise_or(np.left_shift(np.bitwise_and(base, mask), i), multimask_image)
    return multimask_image


def retain_mask_list(multi_mask_image: np.ndarray) -> List[np.ndarray]:
    """ Takes 16bit image and retrieves binary masks. Assumes that each bit is one mask.

    Args:
        multi_mask_image (np.ndarray): 16 bit image containing N masks

    Returns:
        List[np.ndarray]: N binary images
    """
    mask_list = []
    for i in range(16):
        bitmask = np.left_shift(np.ones(multi_mask_image.shape, dtype=np.uint16), i)
        mask = np.bitwise_and(multi_mask_image, bitmask)
        out = np.zeros(multi_mask_image.shape, dtype=np.uint8)
        out[mask > 0] = 255
        if np.sum(out) == 0:
            break
        mask_list.append(out)
    return mask_list


def export_multimaskimage(multi_mask_image, filename: str):
    if not filename.endswith('.png'):
        filename += '.png'
    cv2.imwrite(filename=filename, img=multi_mask_image)


def plot_rgbt_signals(rgbt_signals, fs, face_regions_names=None):
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))
    sig_len = rgbt_signals[0].shape[0]
    time = np.arange(sig_len) / fs
    for idx in range(len(rgbt_signals)):
        axs[0].plot(time, rgbt_signals[idx][:, 0])
    axs[0].set_ylabel('red')
    for idx in range(len(rgbt_signals)):
        axs[1].plot(time, rgbt_signals[idx][:, 1])
    axs[1].set_ylabel('green')
    for idx in range(len(rgbt_signals)):
        axs[2].plot(time, rgbt_signals[idx][:, 2])
    axs[2].set_ylabel('blue')
    if face_regions_names is not None:
        axs[2].legend(face_regions_names)
    axs[2].set_xlabel('time / s')


def save_ppg_data_mat(output_mat_file, rgbts: List[np.ndarray], ppgs: List[np.ndarray], outconfig: Dict[str, Any], avg_hr_bpm=None, create_mask_image=True):
    if create_mask_image:
        _rgb = outconfig.get('rgb0', None)
        _masks = outconfig.get('masks0', None)
        if _rgb is not None and _masks is not None:
            vis_regions = get_contour_image(_rgb, _masks)
            outconfig["vis_regions"] = vis_regions

    out_dic = {'rgbts': rgbts, 'ppgis': ppgs, "config": outconfig, "avg_hr_bpm": avg_hr_bpm}

    def replace_none(data):
        for k, v in data.items() if isinstance(data, dict) else enumerate(data):
            if v is None:
                data[k] = ''
            elif isinstance(v, (dict, list)):
                replace_none(v)

    replace_none(out_dic)
    savemat(output_mat_file, out_dic)


def load_ppg_data_mat(input_mat_file):
    in_dict = loadmat(input_mat_file, simplify_cells=True)
    rgbts = in_dict["rgbts"]
    ppgs = in_dict.get("ppgis", None)
    if len(rgbts.shape) == 3:
        rgbts = [rgbts[_i].reshape(-1) for _i in range(rgbts.shape[0])]
        if ppgs is not None:
            ppgs = [ppgs[_i].reshape(-1) for _i in range(ppgs.shape[0])]
    fs = in_dict["config"]["fs"]
    config = in_dict["config"]
    return rgbts, ppgs, fs, config


def is_gaps_in_files(folder, number_range=[9, -4], verbose=1):
    import os
    import numpy as np
    files = os.listdir(folder)
    files_short = [f[number_range[0]:number_range[1]] for f in files]
    files_short.sort
    file_nums = [int(f) for f in files_short]
    file_arr = np.array(file_nums)
    indices = np.argwhere(np.diff(file_arr) > 1)
    if verbose > 0:
        print(indices)
    return len(indices) > 0
