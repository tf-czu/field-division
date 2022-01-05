"""
    Evaluation the plot shape based on farm machines traffic
"""

import json
import numpy as np
import cv2 as cv

g_px_size = 0.1  # m


def create_erode_element(working_width):
    diameter = int(round(2 * working_width/g_px_size + 1))
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))


def read_plot_shape(plot_file):
    with open(plot_file) as f:
        dic = json.load(f)
        plot_list = dic["plot"]

        return np.asarray(plot_list, dtype=np.int32)

def rotate_cnt(center, cnt, angle):
    cx, cy = center
    angle = np.deg2rad(angle)
    ret = np.zeros(cnt.shape)
    ret[:, 0, 0] = cx + np.cos(angle) * (cnt[:, 0, 0] - cx) - np.sin(angle) * (cnt[:, 0, 1] - cy)
    ret[:, 0, 1] = cy + np.sin(angle) * (cnt[:, 0, 0] - cx) + np.cos(angle) * (cnt[:, 0, 1] - cy)
    ret = ret.round()

    return ret.astype(np.int32)


def extend_traffic_pattern_v1(traffic_pattern, working_width_px, num_ride_around):
    # Extend the pattern in x-axes
    dilate_element = np.ones((1, 2 * num_ride_around * working_width_px + 1))
    print(dilate_element.shape)
    return cv.dilate(traffic_pattern, dilate_element)


def extend_traffic_pattern_v2(traffic_pattern, plot_no_edges, whole_plot):
    traffic_pattern = np.logical_and(traffic_pattern, plot_no_edges).astype(np.uint8)
    dilate_element = np.ones((1, 10 + 1))
    while True:
        extend_pattern = cv.dilate(traffic_pattern,dilate_element)
        new_traffic_pattern = np.logical_and(extend_pattern, whole_plot).astype(np.uint8)
        if np.array_equal(new_traffic_pattern, traffic_pattern):
            return traffic_pattern
        else:
            traffic_pattern = new_traffic_pattern

def eval_plot_shape(plot_cnt, working_width = 4.0, num_ride_around = 2, debug = False):
    if np.min(plot_cnt) < 1:  # too close to edge
        plot_cnt = plot_cnt + 1
    plot_area = cv.contourArea(plot_cnt)
    size = int(np.max(plot_cnt) * 1.5)
    background = np.zeros((size, size), dtype=np.uint8)
    plot_im1 = cv.drawContours(background.copy(), [plot_cnt], -1, color=(255), thickness=-1)

    element = create_erode_element(working_width)
    plot_im2 = cv.erode(plot_im1, element, iterations=num_ride_around)
    contours, hierarchy = cv.findContours(plot_im2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Test result after erode operation. Expected just one object.
    assert len(contours) != 0, "The plot is too small."
    assert len(contours) == 1, "More complex objects are not supported now."
    plot_cnt2 = contours[0]

    # bounding rectangle with minimum area
    (cx, cy), (w, h), angle = cv.minAreaRect(plot_cnt2)
    print(cx, cy, w, h, angle)
    angle = - angle  # https://theailearner.com/tag/cv2-minarearect/
    if h > w:
        angle = angle + 90

    # center objects in image
    dx = int(round(size/2 - cx))  # dist to move in x
    dy = int(round(size/2 - cy))  # dist to move in x
    plot_cnt = plot_cnt + [dx, dy]
    plot_cnt2 = plot_cnt2 + [dx, dy]

    # rotate objects
    plot_cnt = rotate_cnt((size//2, size//2), plot_cnt, angle)
    plot_cnt2 = rotate_cnt((size // 2, size // 2), plot_cnt2, angle)

    # draw whole plot and plot without edges
    whole_plot = cv.drawContours(background.copy(), [plot_cnt], -1, color=(255), thickness=-1)
    plot_no_edges = cv.drawContours(background.copy(), [plot_cnt2], -1, color=(255), thickness=-1)

    # bounding rectangle and draw traffic lines/rectangle
    x, y, w, h = cv.boundingRect(plot_cnt2)
    # print(x, y, w, h)

    # # Ignore separate traffic lines
    # # First extend rectangle to headlands
    # traffic_rec = cv.rectangle(background.copy(), (x, y), (x+w - 1, y+h - 1), (255), -1)
    # # Limit the traffic rectangle to plot only
    # # traffic_pattern = np.logical_and(traffic_rec, plot_no_edges).astype(np.uint8)
    # # extend the pattern to headlands
    # dist = 1 + 2 * int(round(num_ride_around * working_width / g_px_size))
    # dilate_element = np.ones((1, dist))
    # while True:
    #     extend_pattern = cv.dilate(traffic_pattern,dilate_element)
    #     new_traffic_pattern = np.logical_and(extend_pattern, whole_plot).astype(np.uint8)
    #     if np.array_equal(new_traffic_pattern, traffic_pattern):
    #         break
    #     else:
    #         traffic_pattern = new_traffic_pattern

    # Work with each traffic lines
    # First number of traffic lines
    working_width_px = int(round(working_width/g_px_size))
    num_traffic_lines = int(h//(working_width_px) + (1 if h%working_width_px else 0))
    if debug:
        print("num_traffic_lines", num_traffic_lines)
    # Empty traffic pattern
    traffic_pattern = background.copy()
    debug_lines = background.copy().astype(bool)
    # Draw individual traffic lines
    for ii in range(num_traffic_lines):
        # in the beginning, the traffic line is a strip across the whole picture. yl_1 and yl_2 are y-coordinates
        yl_1 = y + ii*working_width_px
        yl_2 = yl_1 + working_width_px
        strip = np.ones((working_width_px, size), dtype=np.uint8)*255
        # The strip is cut according to plot_cnt2
        plot_no_edges_strip = plot_no_edges[yl_1:yl_2, :]
        plot_in_strip = np.any(plot_no_edges_strip, axis=0)
        if debug:
            debug_lines[yl_1 + working_width_px//2 - 1, :] = plot_in_strip
            debug_lines[yl_1 + working_width_px//2, :] = plot_in_strip  # more distinctive lines
        traffic_line = np.logical_and(strip, plot_in_strip).astype(np.uint8)
        traffic_pattern[yl_1:yl_2, :] = traffic_line

    traffic_pattern = extend_traffic_pattern_v1(traffic_pattern, working_width_px, num_ride_around)
    # traffic_pattern = extend_traffic_pattern_v2(traffic_pattern, plot_no_edges, whole_plot)
    # Cut the result
    # headlands = np.logical_and(np.logical_and(whole_plot, np.logical_not(plot_no_edges)), traffic_pattern)
    headlands = np.logical_and(traffic_pattern, np.logical_not(plot_no_edges))
    headlands_area = np.count_nonzero(headlands)
    headlands_num = headlands_area / plot_area * 100  # %

    if debug:
        fig_plot = np.ones((size, size, 3), dtype=np.uint8)*255
        fig_plot[headlands] = [0, 0, 200]
        fig_plot[debug_lines] = [0, 200, 0]
        cv.drawContours(fig_plot, [plot_cnt], -1, color=(0, 0, 0), thickness=2)
        cv.drawContours(fig_plot, [plot_cnt2], -1, color=(200, 0, 0), thickness=2)

        cv.namedWindow('win', cv.WINDOW_NORMAL)
        cv.resizeWindow('win', 800, 800)
        cv.imshow('win', fig_plot)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return headlands_num


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plotfile', help='json file with a plot cnt')
    parser.add_argument('--working_width', help='Working width in m (default 4.0 m)', type=float, default=4.0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    plot_file = args.plotfile
    working_width = args.working_width
    debug = args.debug

    plot = read_plot_shape(plot_file)
    headlands_num = eval_plot_shape(plot, working_width=working_width, debug=debug)
    print("Headlands_num: %0.3f" % headlands_num)
