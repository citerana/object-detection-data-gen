# flake8: noqa
from os.path import join, basename, splitext
from random import randrange
import csv
import glob

import numpy as np
# import pandas as pd
import rasterio
import json
import gdal
import struct

from data_extent import data, folders


def gdal_transform(src_filename, ogr_filename):

    pix = []
    src_ds = gdal.Open(src_filename)
    # = rasterio.open(src_filename)
    gt = src_ds.GetGeoTransform()
    rb = src_ds.GetRasterBand(1)

    ds = ogr.Open(ogr_filename)
    lyr = ds.GetLayer()
    for feat in lyr:
        geom = feat.GetGeometryRef()
        mx, my = geom.GetX(), geom.GetY()  # coord in map units

        # Convert from map to pixel coordinates.
        # Only works for geotransforms with no rotation.
        px = int((mx - gt[0]) / gt[1]) # x pixel
        py = int((my - gt[3]) / gt[5]) # y pixel
        pix.append([px, py])

    return pix


def get_image_corners(corners):
    return [
        min(x[0] for x in corners), # minX
        max(x[0] for x in corners), # maxX
        min(x[1] for x in corners), # minY
        max(x[1] for x in corners) # maxY
    ]


def coordinate_width_and_height(corners):
    return corners[1]-corners[0], corners[3]-corners[2]


def points_from_rectangle(rectangle, maxXcoord, minYcoord):
    rectangle_js = rectangle['geometry']['coordinates'][0]
    return [
        min(maxXcoord - rectangle_js[0][0], maxXcoord - rectangle_js[1][0]), # minX/left
        max(maxXcoord - rectangle_js[0][0], maxXcoord - rectangle_js[1][0]), # maxX/right
        min(rectangle_js[0][1] - minYcoord, rectangle_js[2][1] - minYcoord), # minY/top
        max(rectangle_js[0][1] - minYcoord, rectangle_js[2][1] - minYcoord) # maxY/bottom
    ]


def scale_to_pixel(rect, height_scale, width_scale):
    pixels = []
    pixels.append(rect[0]*width_scale)
    pixels.append(rect[1]*width_scale)
    pixels.append(rect[2]*height_scale)
    pixels.append(rect[3]*height_scale)

    return [int(pixel) for pixel in pixels]


def window_ordered_coords(rasterio_bbox):
    return ((rasterio_bbox[3], rasterio_bbox[1]), (rasterio_bbox[0], rasterio_bbox[2]))


def bbox_ordered_coords(rect):
    return rect[0], rect[3], rect[1], rect[2]


def convert_to_pixels(src_ds, ogr_filename, img_corners):
    pix = []
    img_h, img_w = src_ds.shape[0], src_ds.shape[1]
    with open(ogr_filename) as ogr:
        ogr_dict = json.load(ogr)

    image_corners = get_image_corners(img_corners)
    coord_w, coord_h = coordinate_width_and_height(image_corners)
    height_scale, width_scale = img_h/coord_h, img_w/coord_w

    zeroed_coords = [points_from_rectangle(feat, image_corners[1], image_corners[2])
                    for feat in ogr_dict['features']]
    pixel_coords = [scale_to_pixel(coords, height_scale, width_scale)
                    for coords in zeroed_coords]

    return pixel_coords


def check_in_bbox(x, y, bbox):
    if (x > bbox[0] and x <= bbox[1]) and (y > bbox[2] and y <= bbox[3]):
        return True
    return False


def contains(big, sml):
    if big[0] <= sml[0] and big[1] >= sml[1] and big[2] >= sml[2] and \
       big[3] <= sml[3]:
       return True
    return False


def expand_window(bbox):
    while True:
        ul_X = randrange(bbox[0], bbox[1] + 1)
        ul_Y = randrange(bbox[2], bbox[3] + 1)
        # Check if proposed upper left corner of window is in the bounding box
        if not check_in_bbox(ul_X, ul_Y, bbox):
            break

    return ul_X, ul_X + 256, ul_Y, ul_Y + 256


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


def write_chip_and_ships_to_file(chip_ind, mask, window_box, ship_boxes, chip_ships_dict):
    pass


def generate_chips():
    dataset_path = '/home/annie/Data/datasets/planet_ships/singapore'
    csv_fields = ['coordinates']
    chip_ships_dict = {}
    chip_ind = 0

    for img_ind, folder_name in enumerate(folders[0]):
        folder_path = join(dataset_path, folder_name) + '/'
        img_name = data[img_ind][0]
        img_corners = data[img_ind][1]
        csv_name = 'ships_' + str(img_ind) + '.csv'
        src_path = folder_path + img_name
        ogr_path = folder_path + 'ships_ogr_' + str(img_ind) + '.geojson'

        with rasterio.open(src_path) as src_ds:
            # b, g, r, _ = (src_ds.read(k) for k in (1, 2, 3))
            b, g, r, _ = src_ds.read()

        bbox_coords = convert_to_pixels(src_ds, ogr_path, img_corners)
        ship_boxes = [rasterio.coords.BoundingBox(*bbox_ordered_coords(bbox))
                      for bbox in bbox_coords]
        chip_coords = [expand_window(bbox) for bbox in bbox_coords]

        windows = create_valid_windows(chip_coords, ship_boxes)
#        window_mats = [src_ds.read(1, window=window) for window in windows]
        masks = [window_ordered_coords(window) for window in windows]
#        print(window_mats[0])
        print(masks[0])
        print(src_ds.shape)
        print(src_path)
        for mask, window_box in zip(masks, windows):
            ships_in_chip = []
            for ship in ship_boxes:
                if not rasterio.coords.disjoint_bounds(window_box, ship):
                    ships_in_chip.append(ship[0], ship[2], ship[3], ship[1])

            chip_ships_dict[chip_ind] = ships_in_chip
            write_path = dataset_path + '/train/' + str(chip_ind) + '.jpg'
            with rasterio.open(
                    write_path, 'w',
                    driver='jpeg', width=256, height=256, count=3,
                    dtype='uint8') as dst:
                for k, arr in [(1, b), (2, g), (3, r)]:
                    dst.write(src_ds, window=mask, indexes=3)
                dst.close()
            # write_chip_and_ships_to_file(chip_ind, mask, window_box, \
            # ship_boxes, chip_ships_dict)


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

    #print(gdal_transform(name, ogr_name))
    print(transform(src_name, ogr_name))
    # with rasterio.open(folder_path + '20170622_095940_0c42_3B_AnalyticMS (copy).tif') as dst:
    #     mask = rasterio.features.rasterize(ogr_name, dst.shape)

    # window_corners = [window_ordered_coords(chip) for chip in chip_coords]
    # bbox_corners = [window_ordered_coords(rect) for rect in pixel_coords]
    # windows = [src_ds.window(bbox) for *bbox in bboxes]
    """

    generate_chips()

main()
