"""
This script contains functions to execute histogram based skin segmentation.

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from numba import jit

import roi_segmentation.DEFINITION_FACEMASK


def generate_face_mask(face_landmarks, frame):
    # Create a black mask image
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8) # np.zeros_like(frame)

    # Draw each triangle as white on the mask
    for triangle in roi_segmentation.DEFINITION_FACEMASK.FACE_MESH_TESSELATION:
        pts = np.array([[landmark.x * frame.shape[1], landmark.y * frame.shape[0]]
                        for landmark in face_landmarks.landmark])
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        triangle_pts = [pts[pt] for pt in triangle]
        cv2.fillConvexPoly(mask, np.array(triangle_pts), (255, 255, 255, cv2.LINE_AA))

    return mask


def skin_segmentation(frame):
    # Convert frame to rg-color space
    normalized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0

    # Convert frame to YCrCb color space
    # normalized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    r_channel = normalized_frame[:, :, 0]
    g_channel = normalized_frame[:, :, 1]
    b_channel = normalized_frame[:, :, 2]

    # Calculate 2D histograms for R, G, and B channels
    bins_b, bins_g, bins_r, hist_b, hist_g, hist_r = calc_histograms(r_channel, g_channel, b_channel, n_bins=64)

    lower_threshold_r, upper_threshold_r, lower_threshold_g, upper_threshold_g, lower_threshold_b, upper_threshold_b = \
        calc_histogram_tresholds(hist_r, hist_g, hist_b, bins_r, bins_g, bins_b)

    skin_mask = cv2.inRange(frame, np.array([int(lower_threshold_b*255), int(lower_threshold_g*255), 0]), np.array([255, 255, 255]))

    skin_mask = cv2.medianBlur(skin_mask, 3)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    return skin_mask


@jit(nopython=True)
def calc_histograms(r_channel, g_channel, b_channel, n_bins=64):
    hist_r, bins_r = np.histogram(r_channel.flatten(), bins=n_bins, range=(0, 1))
    hist_g, bins_g = np.histogram(g_channel.flatten(), bins=n_bins, range=(0, 1))
    hist_b, bins_b = np.histogram(b_channel.flatten(), bins=n_bins, range=(0, 1))

    return bins_b, bins_g, bins_r, hist_b, hist_g, hist_r


@jit(nopython=True)
def calc_quartiles(bins, hist):
    # Calculate the cumulative sum of the histogram counts
    cumulative_counts = np.cumsum(hist)

    # Calculate the total count of elements in the histogram
    total_count = cumulative_counts[-1]

    # Calculate the percentile indices corresponding to the quartiles
    lower_quartile_index = np.argmax(cumulative_counts >= total_count * 0.25)
    upper_quartile_index = np.argmax(cumulative_counts >= total_count * 0.75)
    median_index = np.argmax(cumulative_counts >= total_count * 0.5)

    # Calculate the values of the quartiles using the bin edges
    lower_quartile = bins[lower_quartile_index]
    upper_quartile = bins[upper_quartile_index]
    median = bins[median_index]

    return median, lower_quartile, upper_quartile


@jit(nopython=True)
def calc_histogram_tresholds(hist_r, hist_g, hist_b, bins_r, bins_g, bins_b):
    # Find the bin with the maximum count for each channel.
    # The first bin is neglected, as it contains all black pixels
    # The last bin is neglected, as it contains all saturated white pixels
    max_bin_r = np.argmax(hist_r[1:-1]) + 1
    max_bin_g = np.argmax(hist_g[1:-1]) + 1
    max_bin_b = np.argmax(hist_b[1:-1]) + 1

    median_r, lower_quartile_r, upper_quartile_r = calc_quartiles(bins_r[1:-1], hist_r[1:-1])
    median_g, lower_quartile_g, upper_quartile_g = calc_quartiles(bins_g[1:-1], hist_g[1:-1])
    median_b, lower_quartile_b, upper_quartile_b = calc_quartiles(bins_b[1:-1], hist_b[1:-1])

    # Define the range to include the maximum bin and neighboring bins
    bin_range = 2  # int(0.10 * len(bins_r))  # Adjust this range as needed

    # Define the threshold range for skin color for each channel
    threshold_range = 0  # .15  # Adjust this threshold as needed

    # Calculate the lower and upper thresholds for R, G, and B channels
    upper_threshold_r = upper_quartile_r
    upper_threshold_g = upper_quartile_g
    upper_threshold_b = upper_quartile_b

    lower_threshold_r = max(bins_r[max(0, max_bin_r - bin_range)] - threshold_range, 0) \
        if max_bin_r-bin_range < 0.1*len(bins_r) or bins_r[max_bin_r-bin_range] < lower_quartile_r else lower_quartile_r
    lower_threshold_g = max(bins_g[max(0, max_bin_g - bin_range)] - threshold_range, 0) \
        if max_bin_g-bin_range < 0.1*len(bins_g) or bins_g[max_bin_g-bin_range] < lower_quartile_g else lower_quartile_g
    lower_threshold_b = max(bins_b[max(0, max_bin_b - bin_range)] - threshold_range, 0) \
        if max_bin_b-bin_range < 0.1*len(bins_b) or bins_b[max_bin_b-bin_range] < lower_quartile_b else lower_quartile_b

    return lower_threshold_r, upper_threshold_r, lower_threshold_g, upper_threshold_g, lower_threshold_b, upper_threshold_b
