# flake8: noqa
from os.path import join
from random import randrange
import csv

import numpy as np
import pandas as pd
import rasterio
import json
import gdal

from draw_bboxes import draw_bounding_boxes_on_image
from data_extent import data, folders


def get_image_corners(x):
    return [
        x[0][0], #minX
        x[2][0],
        x[1][1], #minY
        x[0][1]
    ]


def coordinate_width_and_height(corners):
    return corners[1]-corners[0], corners[3]-corners[2]


def points_from_rectangle(rectangle, maxXcoord, minYcoord):
    rectangle_js = rectangle['geometry']['coordinates'][0]
    return [
        min(maxXcoord - rectangle_js[0][0], maxXcoord - rectangle_js[1][0]), # minX/left
        max(maxXcoord - rectangle_js[0][0], maxXcoord - rectangle_js[1][0]), # maxX/right
        min(rectangle_js[0][1] - minYcoord, rectangle_js[2][1] - minYcoord), # minY/bottom
        max(rectangle_js[0][1] - minYcoord, rectangle_js[2][1] - minYcoord) # maxY/top
    ]


def scale_to_pixel(rect, height_scale, width_scale):
    pixels = []
    pixels.append(rect[0]*width_scale)
    pixels.append(rect[1]*width_scale)
    pixels.append(rect[2]*height_scale)
    pixels.append(rect[3]*height_scale)

    return [int(pixel) for pixel in pixels]


def percentiles_from_rectangle(rectangle, maxXcoord, maxYcoord, w, h):
    rectangle_js = rectangle['geometry']['coordinates'][0]
    return [
        min(maxXcoord - rectangle_js[0][0], maxXcoord - rectangle_js[1][0])/w, # minX/left
        max(maxXcoord - rectangle_js[0][0], maxXcoord - rectangle_js[1][0])/w, # maxX/right
        min(maxYcoord - rectangle_js[0][1], maxYcoord - rectangle_js[2][1])/h, # minY/bottom
        max(maxYcoord - rectangle_js[0][1], maxYcoord - rectangle_js[2][1])/h, # maxY/top
    ]


def match_percent_to_pixel(rect, img_w, img_h):
    pixels = []
    pixels.append(rect[0]*img_w)
    pixels.append(rect[1]*img_w)
    pixels.append(rect[2]*img_h)
    pixels.append(rect[3]*img_h)

    return [int(pixel) for pixel in pixels]


def convert_to_pixels(src_ds, ogr_filename, img_corners):
    import geopyspark as gps
    from pyspark import SparkContext
    import numpy as np
    import os
    from datetime import datetime
    from shapely.geometry import mapping, shape, Point
    import pyproj
    from shapely.ops import transform
    from functools import partial
    import urllib.request, json
    from geonotebook.wrappers import TMSRasterData
    from PIL import Image

    p = Point(-75.160674, 39.968145)
    project = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:4326'),
            pyproj.Proj(init='epsg:3857'))
    p2 = transform(project, p)
    logic(p2.x, p2.y)

    """
    img_h, img_w = src_ds.shape[0], src_ds.shape[1]
    with open(ogr_filename) as ogr:
        ogr_dict = json.load(ogr)

    image_corners = get_image_corners(img_corners)
    coord_w, coord_h = coordinate_width_and_height(image_corners)

    # Uses scaling up to convert from coords to pixels
    height_scale, width_scale = img_h/coord_h, img_w/coord_w
    zeroed_coords = [points_from_rectangle(feat, image_corners[1], image_corners[2])
                    for feat in ogr_dict['features'][::2]]
    pixel_coords = [scale_to_pixel(coords, height_scale, width_scale)
                    for coords in zeroed_coords]

    # Uses a percentile location to convert across units
    percentile_coords = [percentiles_from_rectangle(feat, image_corners[1], image_corners[3], coord_w, coord_h)
                         for feat in ogr_dict['features'][::2]]
    #print(img_w, img_h)
    #print(percentile_coords)
    pixel_coords2 = [match_percent_to_pixel(coords, img_w, img_h) for coords in percentile_coords]
    #print(pixel_coords2)

    return pixel_coords2
    """

def window_ordered_coords(rasterio_bbox):
    # ymin, ymax, xmin, xmax
    return ((rasterio_bbox[3], rasterio_bbox[1]), (rasterio_bbox[0], rasterio_bbox[2]))


def bbox_ordered_coords(rect):
    # xmin, ymin, xmax, ymax
    return rect[0], rect[3], rect[1], rect[2]


def tf_ordered_coords(rect):
    # ymin, xmin, ymax, xmax
    return [rect[2], rect[0], rect[3], rect[1]]


def check_in_bbox(x, y, bbox):
    if (x > bbox[0] and x <= bbox[1]) and (y > bbox[2] and y <= bbox[3]):
        return True
    return False


def contains(big, sml):
    if big[0] <= sml[0] and big[1] >= sml[1] and big[2] >= sml[2] and \
       big[3] <= sml[3]:
       return True
    return False


def expand_window_with_offset(bbox):
    while True:
        ul_X = randrange(bbox[0], bbox[1] + 1)
        ul_Y = randrange(bbox[2], bbox[3] + 1)
        # Check if proposed upper left corner of window is in the bounding box
        if not check_in_bbox(ul_X, ul_Y, bbox):
            break

    return ul_X, ul_X + 256, ul_Y, ul_Y + 256


def expand_window_no_offset(bbox):
    if bbox[0] - 128 < 0 or bbox[0] + 128 < 0 or bbox[1] - 128 < 0 or bbox[1] + 128 < 0:
        return 0, 256, 0, 256
    return bbox[0] - 128, bbox[0] + 128, bbox[1] - 128, bbox[1] + 128


def create_valid_windows(chips, ship_boxes):
    windows = []
    for chip in chips:
        valid_window = True
        window_box = rasterio.coords.BoundingBox(*bbox_ordered_coords(chip))
        for ship in ship_boxes:
            # Check if ship overlaps with the window and if it does
            # check that the ship is completely in the window.
            if not rasterio.coords.disjoint_bounds(window_box, ship):
                if not contains(window_box, ship):
                    valid_window = False
        if valid_window:
            windows.append(window_box)

    return windows


def generate_chips():
    dataset_path = '/home/annie/Data/datasets/planet_ships/singapore'
    chip_ships_list = []
    chip_ind = 0

    for img_ind, folder_name in enumerate(folders):
        folder_path = join(dataset_path, folder_name) + '/'
        img_name = data[img_ind][0]
        img_corners = data[img_ind][1]
        csv_name = 'ships_' + str(img_ind) + '.csv'
        src_path = folder_path + img_name
        ogr_path = folder_path + 'ships_ogr_' + str(img_ind) + '.geojson'

        with rasterio.open(src_path) as src_ds:
            b, g, r, ir = (src_ds.read(k) for k in (1, 2, 3, 4))

            bbox_coords = convert_to_pixels(src_ds, ogr_path, img_corners)
            ship_boxes = [rasterio.coords.BoundingBox(*bbox_ordered_coords(bbox))
                          for bbox in bbox_coords]
            #chip_coords = [expand_window_with_offset(bbox) for bbox in bbox_coords]
            chip_coords = [expand_window_no_offset(bbox) for bbox in bbox_coords]

            windows = create_valid_windows(chip_coords, ship_boxes)
            masks = [window_ordered_coords(window) for window in windows]

            for mask, window_box in zip(masks, windows):
                for ship in ship_boxes:
                    if not rasterio.coords.disjoint_bounds(window_box, ship):
                        chip_ships_list.append((str(chip_ind), ship[0], ship[1], ship[2], ship[3]))
                img_write_path = dataset_path + '/train/' + str(chip_ind) + '.tif'
                with rasterio.open(
                        img_write_path, 'w',
                        driver='GTiff', width=256, height=256, count=3,
                        dtype='uint16') as dst:
                    dst.write(src_ds.read(1, window=mask), indexes=1) #b
                    dst.write(src_ds.read(2, window=mask), indexes=2) #g
                    dst.write(src_ds.read(3, window=mask), indexes=3) #r
                    #     img_write_path, 'w',
                    #     driver='png', width=256, height=256, count=3,
                    #     dtype='uint8') as dst:
                    # dst.write(src_ds.read(3, window=mask).astype(np.uint8), indexes=1) #b
                    # dst.write(src_ds.read(2, window=mask).astype(np.uint8), indexes=2) #g
                    # dst.write(src_ds.read(1, window=mask).astype(np.uint8), indexes=3) #r
                    # dst.close()
                chip_ind += 1
                print(str(chip_ind) + " chip written")
        raw_input("Continue? ")
    # Once all chips have been created, write ship loc dataframe to csv
    print(chip_ships_list)
    csv_labels = ['image', 'ship (l, t, r, b)']
    df = pd.DataFrame.from_records(chip_ships_list, columns=csv_labels)
    csv_write_path = dataset_path + '/train/' + 'ship_locations.csv'
    df.to_csv(csv_write_path, index=False)


def draw_bboxes_on_scenes():
    _path = '/home/annie'

    for img_ind, folder_name in enumerate(folders):
        folder_path = join(dataset_path, folder_name) + '/'
        img_name = data[img_ind][0]
        img_corners = data[img_ind][1]
        src_path = dataset_path + '/' + str(img_ind) + '.png'
        ogr_path = folder_path + 'ships_ogr_' + str(img_ind) + '.geojson'
        img_write_path = '/home/annie/' + str(img_ind) + '.png'


        with rasterio.open(src_path) as src_ds:
            b, g, r, ir = (src_ds.read(k) for k in (1, 2, 3, 4))
            pixel_coords = convert_to_pixels(src_ds, ogr_path, img_corners)
            bbox_coords = np.array([tf_ordered_coords(coords) for coords in pixel_coords])
            print(bbox_coords)
            img_boxes = draw_bounding_boxes_on_image(src_path, bbox_coords)
            img_boxes.save(write_path, "PNG")
            # with rasterio.open(
            #         img_write_path, 'w',
            #         driver='GTiff', width=src_ds.shape[1], height=src_ds.shape[0],
            #         count=3, dtype='uint16') as dst:
            #     dst.write(src_ds.read(1), indexes=1) #b
            #     dst.write(src_ds.read(2), indexes=2) #g
            #     dst.write(src_ds.read(3), indexes=3) #r


def main():
    """
    for each scene
        load .csv of bounding boxes into pandas
        for each row in the bounding boxes of an image,
            pick a window with some random offset
                check if the edges intersect with another boxes
                    if so, discard this box
                otherwise, convert the BB geo-location wrt .png chip pixels
                check if the image has gone off the edge of the scene
                    if so, convert blank areas into neutral color
                    map all known ships in chip to a CSV
        df_path = join(folder_path, csv_name)
        bbdf = pd.read_csv(df_path, skipinitialspace=True, usecols=csv_fields)
        # we're only interested in every other line of the csv
        for bb in bbdf.values[::2]:
            with rasterio.open("tests/data/RGB.byte.tif") as src:
                mask = rasterio.features.rasterize(ogr_name, src.shape)

    folder_path = '/home/annie/Data/datasets/planet_ships/singapore/20170622_095940_0c42/'
    ogr_name = folder_path + 'ships_ogr_5.geojson'
    src_name = folder_path + '20170622_095940_0c42_3B_AnalyticMS.tif'
    name = folder_path + 'TC_NG_Baghdad_IQ_Geo.tif'
    """

    draw_bboxes_on_scenes()
    # generate_chips()

main()
