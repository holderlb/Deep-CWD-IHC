# contour-region-image.py --image <svs_slide_file> --contour <contour_file> --regions <regions_file>
#   --tiles_dir <tiles_directory> --probability <probability>
#
# Generate image of contour with regions highlighted and write to <image>-regions.png. Also generate
# image highlighting tiles with P(class) > probability and write to <image>-regions-prob.png. Also
# generate image with all tiles shaded according to P(class) and write to <image>-regions-heatmap.png.
#
# Author: Lawrence Holder, Washington State University

import argparse
from openslide import OpenSlide
from PIL import Image, ImageDraw
import json
import numpy as np
import skimage
import os
import csv

SCALE_FACTOR = 32
TILES_CSV_FILE = 'tiles.csv'
POSITIVE_CLASS_NAMES = ['follicle', 'dorsal_motor_nucleus'] # positive classes to look for

def scale_image(slide):
    w, h = slide.dimensions
    new_size = (w / SCALE_FACTOR, h / SCALE_FACTOR)
    img = slide.get_thumbnail(size=new_size)
    return img

def add_regions(image, geojson_file_name):
    draw = ImageDraw.Draw(image)
    color_rgb = (0,0,0) # default black
    with open(geojson_file_name, 'r') as geof:
        features = json.load(geof)
        for feature in features:
            geometry = feature['geometry']
            properties = feature['properties']
            classification = properties['classification']
            color_rgb = tuple(classification['color'])
            coordinates = geometry['coordinates'][0]
            points = []
            for point in coordinates:
                x = point[0] // SCALE_FACTOR
                y = point[1] // SCALE_FACTOR
                points.append((x,y))
            draw.polygon(points, outline=color_rgb, width=2)
    return color_rgb

def add_midline(image, contour_file):
    draw = ImageDraw.Draw(image)
    color_rgb = (0,0,0) # default black
    with open(contour_file, 'r') as geof:
        features = json.load(geof)
        for feature in features:
            geometry = feature['geometry']
            properties = feature['properties']
            classification = properties['classification']
            color_rgb = tuple(classification['color'])
            if classification['name'].lower() == 'midline':
                coordinates = geometry['coordinates']
                points = []
                for point in coordinates:
                    x = point[0] // SCALE_FACTOR
                    y = point[1] // SCALE_FACTOR
                    points.append((x,y))
                draw.line([points[0],points[-1]], fill=color_rgb, width=5)
    return

def read_and_scale_contour(contour_file_name):
    with open(contour_file_name, 'r') as contour_file:
        features = json.load(contour_file)
        contour = []
        for feature in features:
            geometry = feature['geometry']
            if geometry['type'].lower() == 'polygon':
                coordinates = geometry['coordinates'][0]
                for point in coordinates:
                    x = point[0] // SCALE_FACTOR
                    y = point[1] // SCALE_FACTOR
                    contour.append((y,x))
    return contour

def extract_contour_image(image, contour):
    """Return portion of image outlined by contour as a square image with white background."""
    # Compute image mask based on contour
    rgb_mask = np.zeros_like(image) # Start with black background
    contour1 = np.round(contour).astype(int)
    rr, cc = skimage.draw.polygon(contour1[:, 0], contour1[:, 1])
    rgb_mask[rr, cc] = [1,1,1]
    masked_img = np.multiply(image, rgb_mask)
    # Extract bounding box from masked image
    min_row, min_col = np.min(rr), np.min(cc)
    max_row, max_col = np.max(rr)+1, np.max(cc)+1
    bb_img = masked_img[min_row:max_row, min_col:max_col]
    # Square off bounding box by adding black rows/cols
    side_length = max(max_row - min_row, max_col - min_col)
    if (max_row - min_row) < side_length:
        # Add rows of black to top and bottom of image
        n_top_rows = (side_length - (max_row - min_row)) // 2
        top_rows = np.zeros((n_top_rows, bb_img.shape[1], bb_img.shape[2]), dtype=image.dtype)
        bb_img = np.vstack((top_rows, bb_img))
        n_bottom_rows = (side_length - (max_row - min_row) - n_top_rows)
        bottom_rows = np.zeros((n_bottom_rows, bb_img.shape[1], bb_img.shape[2]), dtype=image.dtype)
        bb_img = np.vstack((bb_img, bottom_rows))
    if (max_col - min_col) < side_length:
        # Add columns of black (0,0,0) to left and right of image
        n_left_cols = (side_length - (max_col - min_col)) // 2
        left_cols = np.zeros((bb_img.shape[0], n_left_cols, bb_img.shape[2]), dtype=image.dtype)
        bb_img = np.hstack((left_cols, bb_img))
        n_right_cols = (side_length - (max_col - min_col) - n_left_cols)
        right_cols = np.zeros((bb_img.shape[0], n_right_cols, bb_img.shape[2]), dtype=image.dtype)
        bb_img = np.hstack((bb_img, right_cols))
    # Finally convert black background to white
    black_pixels = np.all(bb_img == [0, 0, 0], axis=-1)
    bb_img[black_pixels] = [255, 255, 255]
    return bb_img

def add_tiles(image, color_rgb, args, heatmap):
    draw = ImageDraw.Draw(image, 'RGBA')
    tiles_csv_file = os.path.join(args.tiles_dir, TILES_CSV_FILE)
    with open(tiles_csv_file, 'r') as csv_file:
        csv_dict_reader = csv.DictReader(csv_file)
        for row in csv_dict_reader:
            # Read and scale tile coordinates
            x = (int(row['x']) // SCALE_FACTOR)
            y = int(row['y']) // SCALE_FACTOR
            w = int(row['width']) // SCALE_FACTOR
            h = int(row['height']) // SCALE_FACTOR
            rect_coords = (x, y, x+w, y+h)
            # Get P(class)
            class_prob = float(row['feature_probability'])
            if row['feature_prediction'] not in POSITIVE_CLASS_NAMES:
                class_prob = 1.0 - class_prob
            # Add tile info to image
            if (not heatmap) and (class_prob >= args.probability):
                draw.rectangle(rect_coords, outline=color_rgb, width=1)
            if heatmap:
                alpha = int(class_prob * 128)
                fill_color = color_rgb + (alpha,)
                rect_coords = (x, y, x+w, y+h)
                draw.rectangle(rect_coords, fill=fill_color)
    return

def enhance_pil_image(pil_image):
    np_image = np.array(pil_image)
    np_image = skimage.exposure.equalize_adapthist(np_image) # removes alpha channel, returns floats
    np_image = np_image * 255
    np_image = np_image.astype(np.uint8)
    pil_image = Image.fromarray(np_image)
    return pil_image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_file', type=str) 
    parser.add_argument('--contour', dest='contour_file', type=str)
    parser.add_argument('--regions', dest='regions_file', type=str)
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str)
    parser.add_argument('--probability', dest='probability', type=float)
    parser.add_argument('--enhance', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    image_file = args.image_file
    regions_file = args.regions_file
    contour_file = args.contour_file
    regions_file_base = os.path.splitext(regions_file)[0]
    # Read SVS image
    #print("Reading image \"" + image_file + "\"...", flush=True)
    slide = OpenSlide(image_file)
    scaled_pil_image = scale_image(slide)
    if args.enhance:
        scaled_pil_image = enhance_pil_image(scaled_pil_image)
    print(f'Generating images for region {regions_file}...')
    color_rgb = add_regions(scaled_pil_image, regions_file)
    add_midline(scaled_pil_image, contour_file)
    scaled_pil_image_copy = scaled_pil_image.copy()
    # Generate image with just regions
    scaled_np_image = np.array(scaled_pil_image)
    scaled_contour = read_and_scale_contour(contour_file)
    contour_region_image = extract_contour_image(scaled_np_image, scaled_contour)
    outfile = regions_file_base + '.png'
    skimage.io.imsave(outfile, contour_region_image)
    # Generate image with high-probability tiles highlighted
    scaled_pil_image = scaled_pil_image_copy
    scaled_pil_image_copy = scaled_pil_image.copy()
    add_tiles(scaled_pil_image, color_rgb, args, heatmap=False)
    scaled_np_image = np.array(scaled_pil_image)
    scaled_contour = read_and_scale_contour(contour_file)
    contour_region_image = extract_contour_image(scaled_np_image, scaled_contour)
    outfile = regions_file_base + '-prob.png'
    skimage.io.imsave(outfile, contour_region_image)
    # Generate image with heatmap
    scaled_pil_image = scaled_pil_image_copy
    add_tiles(scaled_pil_image, color_rgb, args, heatmap=True)
    scaled_np_image = np.array(scaled_pil_image)
    scaled_contour = read_and_scale_contour(contour_file)
    contour_region_image = extract_contour_image(scaled_np_image, scaled_contour)
    outfile = regions_file_base + '-heatmap.png'
    skimage.io.imsave(outfile, contour_region_image)
    return

if __name__ == "__main__":
    main()