# detect-midline.py --image <svs_slide> --contour_file <contour_file> --obex_template_file <obex_template_file>
#                   --tiles_dir <tiles_dir> [--probability <probability>=0.5]
#
# Detects line segment constituting the midline of an obex contour and writes it
# back to the contour JSON file. There are three methods:
#
# 1. Find line bisecting the largest concavity of the contour.
# 2. Find line fit to tiles classified as midline tiles. If the --probability
#    argument is provided, then the prediction probability must be at least
#    this high (default is 0.5).
# 3. Fit contour in given obex template file to contour and transform template
#    midline accordingly.
#
# Also outputs images for each method's result.
#
# Author: Lawrence Holder, Washington State University

import os
import csv
import sys
import argparse
import json
import scipy
import numpy as np
from shapely.geometry import Point, LineString
from openslide import OpenSlide
from PIL import Image, ImageDraw
import skimage

TILES_CSV_FILE = 'tiles.csv'
MIDLINE_CLASS_NAME = 'midline'
MIDLINE_PREDICTION_COLUMN = 'midline_prediction'
MIDLINE_PROBABILITY_COLUMN = 'midline_probability'
CONTOUR_MIN_POINTS = 50
SCALE_FACTOR = 32
MIDLINE_COLOR = [255,0,255] # fuchsia

def scale_image(slide):
    w, h = slide.dimensions
    new_size = (w / SCALE_FACTOR, h / SCALE_FACTOR)
    img = slide.get_thumbnail(size=new_size)
    return img

def extract_contour_image(image, contour):
    """Return portion of image outlined by contour as a square image with white background."""
    # Compute image mask based on contour
    rgb_mask = np.zeros_like(image) # Start with black background
    contour1 = np.round(contour).astype(int)
    rr, cc = skimage.draw.polygon(contour1[:, 1], contour1[:, 0]) # expects y,x points
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

def enhance_pil_image(pil_image):
    np_image = np.array(pil_image)
    np_image = skimage.exposure.equalize_adapthist(np_image) # removes alpha channel, returns floats
    np_image = np_image * 255
    np_image = np_image.astype(np.uint8)
    pil_image = Image.fromarray(np_image)
    return pil_image

def generate_midline_image(slide, contour, midline, enhance=False):
    scaled_pil_image = scale_image(slide)
    if enhance:
        scaled_pil_image = enhance_pil_image(scaled_pil_image)
    if midline:
        scaled_midline = [(x // SCALE_FACTOR, y // SCALE_FACTOR) for (x,y) in midline]
        draw = ImageDraw.Draw(scaled_pil_image)
        draw.line([scaled_midline[0],scaled_midline[-1]], fill=tuple(MIDLINE_COLOR), width=5)
    scaled_np_image = np.array(scaled_pil_image)
    scaled_contour = [(x // SCALE_FACTOR, y // SCALE_FACTOR) for (x,y) in contour]
    contour_image = extract_contour_image(scaled_np_image, scaled_contour)
    return contour_image

def generate_midline_image_with_tiles(slide, contour, tiles, midline, enhance=False):
    scaled_pil_image = scale_image(slide)
    if enhance:
        scaled_pil_image = enhance_pil_image(scaled_pil_image)
    draw = ImageDraw.Draw(scaled_pil_image)
    # Add tiles
    for tile in tiles:
        (x,y,w,h) = tile
        rect_coords = (x // SCALE_FACTOR, y // SCALE_FACTOR, (x+w) // SCALE_FACTOR, (y+h) // SCALE_FACTOR)
        draw.rectangle(rect_coords, outline=tuple(MIDLINE_COLOR), width=1)
    # Add midline
    if midline:
        scaled_midline = [(x // SCALE_FACTOR, y // SCALE_FACTOR) for (x,y) in midline]
        draw.line([scaled_midline[0],scaled_midline[-1]], fill=tuple(MIDLINE_COLOR), width=5)
    # Extract contour
    scaled_np_image = np.array(scaled_pil_image)
    scaled_contour = [(x // SCALE_FACTOR, y // SCALE_FACTOR) for (x,y) in contour]
    contour_image = extract_contour_image(scaled_np_image, scaled_contour)
    return contour_image

def read_contour(contour_file_name):
    """Reads and returns contour coordinates and contour classification.
    Assumes only one feature."""
    with open(contour_file_name, 'r') as contour_file:
        features = json.load(contour_file)
        contour = []
        class_name = None
        for feature in features:
            geometry = feature['geometry']
            if geometry['type'].lower() == 'polygon':
                coordinates = geometry['coordinates'][0]
                for point in coordinates:
                    x = point[0]
                    y = point[1]
                    contour.append((x,y))
                properties = feature['properties']
                classification = properties['classification']
                class_name = classification['name']
    return contour, class_name

def read_template(contour_file_name):
    """Reads and returns template contour coordinates and template midline coordinates.
    Assumes only one polygon feature and one linestring feature."""
    with open(contour_file_name, 'r') as contour_file:
        features = json.load(contour_file)
        contour = []
        midline = []
        for feature in features:
            geometry = feature['geometry']
            if geometry['type'].lower() == 'polygon':
                coordinates = geometry['coordinates'][0]
                for point in coordinates:
                    x = point[0]
                    y = point[1]
                    contour.append((x,y))
            elif geometry['type'].lower() == 'linestring':
                coordinates = geometry['coordinates']
                for point in coordinates:
                    x = point[0]
                    y = point[1]
                    midline.append((x,y))
            else:
                continue
    return contour, midline

def simplify_contour(contour):
    """Remove every other point until number of points below min points."""
    while len(contour) > CONTOUR_MIN_POINTS:
        contour = contour[::2]
    return contour

def read_tiles(tiles_dir, min_prob):
    tiles_file_name = os.path.join(tiles_dir, TILES_CSV_FILE)
    tiles = []
    with open(tiles_file_name, 'r') as tiles_file:
        for row in csv.DictReader(tiles_file):
            predicted_class = row[MIDLINE_PREDICTION_COLUMN]
            probability = float(row[MIDLINE_PROBABILITY_COLUMN])
            if (predicted_class == MIDLINE_CLASS_NAME) and (probability >= min_prob):
                x = int(row['x'])
                y = int(row['y'])
                w = int(row['width'])
                h = int(row['height'])
                tile = [x, y, w, h]
                tiles.append(tile)
    return tiles

def compute_midline_by_concavity(contour):
    """Return two points defining the midline segment for the given obex contour.
    The first point should be closest to the crevice."""
    contour = simplify_contour(contour)
    concavities = find_concavities(contour)
    max_concavity = find_largest_concavity(contour, concavities)
    midline = compute_concavity_midline(contour, max_concavity)
    return midline

def find_concavities(contour):
    """Return the indices of pairs of points in the contour that represent the
    beginning and end of a concavity."""
    concavities = []
    hull = scipy.spatial.ConvexHull(contour)
    hull_points = hull.vertices
    i = 0
    while i < len(hull_points)-1:
        index1 = hull_points[i]
        index2 = hull_points[i+1]
        if index2 != index1 + 1:
            concavities.append((index1,index2))
        i += 1
    return concavities

def find_largest_concavity(contour, concavities):
    max_area = 0
    max_concavity = None
    for concavity in concavities:
        concavity_contour = create_sub_contour(concavity, contour)
        area = compute_contour_area(concavity_contour)
        if area > max_area:
            max_concavity = concavity
            max_area = area
    return max_concavity

def create_sub_contour(concavity, contour):
    if concavity[0] < concavity[1]:
        sub_contour = contour[concavity[0]:concavity[1]+1]
    else:
        sub_contour = contour[concavity[0]:] + contour[:concavity[1]+1]
    sub_contour.append(sub_contour[0]) # close contour
    return sub_contour

def compute_contour_area(contour):
    contour = np.array(contour)
    x = contour[:,0]
    y = contour[:,1]
    contour_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return contour_area

def compute_concavity_midline(contour, concavity):
    """Finds the line AB made by the two points that comprise the concavity, which is a pair
    of indices into the contour. Then, finds the line CD perpendicular to AB and intersecting
    the point C in the concavity furthest from AB. Returns the line as two points C and D
    intersecting the contour."""
    contour_np = np.array(contour)
    A = contour_np[concavity[0]]
    B = contour_np[concavity[1]]
    # Find the point in the concavity furthest from AB
    C = furthest_point(contour_np, concavity)
    # Calculate the slope of the line segment AB
    denom = B[0] - A[0]
    if denom == 0:
        denom = 0.001
    slope_AB = (B[1] - A[1]) / denom
    # Calculate the slope of the perpendicular midline
    slope_midline = -1 / slope_AB
    # Find another point along the line
    min_x = np.min(contour_np[:, 0])
    max_x = np.max(contour_np[:, 0])
    x2 = int((min_x + max_x) / 2)
    y2 = int(slope_midline * (x2 - C[0]) + C[1])
    D = (x2, y2)
    C = (int(C[0]), int(C[1]))
    return [C,D]

def furthest_point(contour_np, concavity):
    """Find the furthest point in the concavity from the line defined by the end points
    of the concavity in the contour."""
    A = contour_np[concavity[0]]
    B = contour_np[concavity[1]]
    line = LineString([A,B])
    furthest_point = A
    furthest_distance = 0.0
    i = concavity[0]
    while i != concavity[1]:
        point = Point(contour_np[i][0], contour_np[i][1])
        distance = point.distance(line)
        if distance > furthest_distance:
            furthest_distance = distance
            furthest_point = contour_np[i]
        i = (i + 1) % len(contour_np)
    return furthest_point

def add_midline_to_contour(contour_file, midline):
    new_features = []
    with open(contour_file, 'r') as cf:
        features = json.load(cf)
        for feature in features:
            geometry = feature['geometry']
            if geometry['type'].lower() == 'polygon':
                new_features.append(feature)
                line_feature = get_line_feature(midline)
                new_features.append(line_feature)
    with open(contour_file, 'w') as cf:
        json.dump(new_features, cf, indent=2)
    return

def get_line_feature(midline):
    coords = [[p[0],p[1]] for p in midline]
    feature = {"type": "Feature"}
    geometry = {"type": "LineString"}
    geometry["coordinates"] = coords
    feature["geometry"] = geometry
    classification = {"name": "midline", "color": MIDLINE_COLOR}
    properties = {"classification": classification}
    feature["properties"] = properties
    return feature

def compute_midline_by_tiles(tiles, contour):
    """ Compute midline as the linear regressor of the centers of the tiles."""
    if len(tiles) < 2:
        return None
    points = np.array([[(x + (w // 2)), (y + (h // 2))] for [x,y,w,h] in tiles])
    x = points[:, 0]
    y = points[:, 1]
    slope, intercept = np.polyfit(x, y, 1)
    contour_np = np.array(contour)
    x1 = np.min(contour_np[:, 0])
    x2 = np.max(contour_np[:, 0])
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept
    midline = [(x1,y1), (x2,y2)]
    return midline

def compute_midline_by_template(template_contour, template_midline, contour):
    midline = compute_midline_by_concavity(contour) # TODO: Placeholder
    return midline

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_file', type=str, required=True) 
    parser.add_argument('--contour_file', dest='contour_file', type=str, required=True)
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str, required=True)
    parser.add_argument('--obex_template_file', dest='obex_template_file', type=str) 
    parser.add_argument('--probability', dest='min_prob', default=0.5, type=float)
    parser.add_argument('--enhance', action='store_true')
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_arguments()
    image_file_name = args.image_file
    contour_file_name = args.contour_file
    obex_template_file = args.obex_template_file
    tiles_dir = args.tiles_dir
    min_prob = args.min_prob
    contour_base_name = os.path.splitext(contour_file_name)[0]
    contour, contour_class_name = read_contour(contour_file_name)
    if contour_class_name != 'obex':
        print(f'Contour {contour_file_name} is not classified as obex')
        sys.exit()
    slide = OpenSlide(image_file_name)
    # Method 1: concavity
    midline1 = compute_midline_by_concavity(contour)
    image = generate_midline_image(slide, contour, midline1, args.enhance)
    outfile = contour_base_name + '-midline-concavity.png'
    skimage.io.imsave(outfile, image)
    # Method 2: tiles
    tiles = read_tiles(tiles_dir, min_prob)
    midline2 = compute_midline_by_tiles(tiles, contour)
    image = generate_midline_image_with_tiles(slide, contour, tiles, midline2, args.enhance)
    outfile = contour_base_name + '-midline-tiles.png'
    skimage.io.imsave(outfile, image)
    # Method 3: template
    if obex_template_file:
        obex_template_contour, obex_template_midline = read_template(obex_template_file)
        midline3 = compute_midline_by_template(obex_template_contour, obex_template_midline, contour)
        image = generate_midline_image(slide, contour, midline3, args.enhance)
        outfile = contour_base_name + '-midline-template.png'
        skimage.io.imsave(outfile, image)
    best_midline = midline1 # TODO: which one is best?
    add_midline_to_contour(contour_file_name, best_midline)
    return

if __name__ == "__main__":
    main()
    