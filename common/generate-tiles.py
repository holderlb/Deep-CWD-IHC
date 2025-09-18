# generate-tiles.py --image <svs_slide_file> --annotations <annotations_file>
#                   --tiles_dir <tiles_dir> --overlap <N.N> --tile_size <N>
#                   --tile_increment <N> --enhance
#
# The program generates tiles from the given SVS slide according to the given
# annotations. The <annotations_file> is a GeoJSON-formatted array, e.g., from
# the QuPath program. Each annotation includes "geometry":"coordinates" of the
# points of a set of line segments or of a polygon encompassing a region. The
# "properties":"classification":"name" indicates the class name of the line or
# region, which is later used for learning. If a tile overlaps a line/region,
# then it is written to the directory named after that line/region's class.
# The amount of overlap necessary is controlled by the --tile_overlap float
# option, between 0.0 and 1.0 (default=0.9). For polygon regions, the fraction
# of tile area that overlaps the region must be at least the tile overlap.
# For lines, the tile overlap is ignored, and the tile merely has to intersect
# with the line (but see NOTE 2 below). Tiles are square and their size is
# given by the --tile_size integer option (default=300). The tile_increment
# option defines by how many pixels the tile template is shifted as it passes
# over the image. The default is the tile_size (i.e., tiles don't overlap).
#
# Each class has a corresponding RGB color. These class colors are written
# to <tiles_dir>/colors.json as a dictionary in the form {"<class_name>": [R, G, B]}.
#
# The tile images are stored in the <tiles_dir>/images/<class_name> subdirectories.
# The tile image file name is of the form: <annotations_file>_<NNNNNN>.png,
# where <NNNNNN> is a unique 6-digit, 0-padded number assigned to the tile image.
# The details about the tiles are appended to the file <tiles_dir>/tiles.csv
# (tile image file name, x/y/width/height in image, class name, #stainedpixels).
#
# If the --enhance argument is given, then tiles are enhanced for improved
# contrast using skimage.exposure.equalize_adapthist method.
#
# NOTE 1: This program does not remove existing tiles, will overwrite existing
# tiles with the same name, and will append tile information to an existing
# <tiles_dir>/tiles.csv file. This allows the program to be run multiple times,
# once for each image/annotations, and collect all the tiles in one place,
# if desired.
#
# NOTE 2: Code exists in the intersects_enough method to respect the
# --tile_overlap parameter for lines. The tile_overlap applies to half the tile
# size. So, e.g., if the tile size is 300x300, then the smaller area of the
# intersection of the tile and line must be at lease tile_overlap times
# 300x300x0.5, i.e., if the line halves the tile, then that is equivalent
# to an overlap of 1.0. This approach is commented out in the code due
# to the complexity of determining the correct areas.
#
# Author: Lawrence Holder, Washington State University

import os
import argparse
from openslide import OpenSlide
import json
from shapely.geometry import Polygon, LineString
from shapely.validation import make_valid
from scipy.spatial import ConvexHull
import numpy as np
from skimage import io, exposure

def init_files(tiles_dir):
    # Create tiles and images directories if not already there.
    os.makedirs(tiles_dir, exist_ok=True)
    images_dir = os.path.join(tiles_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    # Create <tiles_dir>/tiles.csv file if not already there.
    tiles_file = os.path.join(tiles_dir, 'tiles.csv')
    if not os.path.exists(tiles_file):
        with open(tiles_file,'w') as tfile:
            tfile.write("image,x,y,width,height,directory,stained_pixels\n")
    return

def write_tile(args, image, num_tiles, tile, dir_name):
    """Write tile image into <tiles_dir>/images/<dir_name> directory. The image file name
    is constructed using the annotations file name as a base, because there may be multiple
    segement contour annotations for each individual image."""
    # Extract elements of image file name
    tiles_dir = args.tiles_dir
    base_file_name = os.path.basename(args.annotations_file)
    base_file_name_noext = os.path.splitext(base_file_name)[0]
    x,y,w,h = tile
    tile_img = image.read_region((x,y), 0, (w,h)) # this is a PIL image
    tile_img = tile_img.convert('RGB') # remove alpha channel if present
    tile_img = np.array(tile_img) # Convert to skimage format
    tile_file_base_name = base_file_name_noext + '_' + str(num_tiles).zfill(6)
    tile_file_name = os.path.join(tiles_dir, 'images', dir_name, tile_file_base_name + '.png')
    num_stained_pixels = count_stained_pixels(tile_img)
    # Enhance tile image
    if args.enhance:
        tile_img = exposure.equalize_adapthist(tile_img) # removes alpha channel, returns floats
        tile_img = tile_img * 255
        tile_img = tile_img.astype(np.uint8)
    # Save tile image
    io.imsave(tile_file_name, tile_img)
    # Append tile information to CSV file
    tiles_file = os.path.join(tiles_dir, 'tiles.csv')
    with open(tiles_file, 'a') as tfile:
        line = tile_file_base_name
        line += ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)
        line += ',' + dir_name
        line += ',' + str(num_stained_pixels)
        tfile.write(line + '\n')
    return

def count_stained_pixels(image, threshold=0.35):
    """Returns number of pixels in image that exhibit staining, i.e.,
    r / (r+g+b) >= threshold. Image is assumed to be RGB in Scikit Image
    format (i.e., NumPy array) with no alpha channel."""
    num_stained_pixels = 0
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            r, g, b = image[y,x].astype(int)
            sum = r + g + b
            if (sum > 0):
                ratio = r / sum
                if ratio >= threshold:
                    num_stained_pixels += 1
    return num_stained_pixels

def parse_qupath_annotations(annotations_file_name):
    """Returns a list of shapes (polygon or line), their class names, and their colors,
    as read from the annotations file."""
    print("Reading annotations...", flush=True)
    with open(annotations_file_name) as annotations_file:
        annotations = json.load(annotations_file)
    shapes = []
    class_names = []
    colors = []
    for annotation in annotations:
        # Check for properly-formatted annotation
        format_okay = True
        if ("geometry" not in annotation) or ("properties" not in annotation):
            format_okay = False
        elif ("type" not in annotation["geometry"]) or ("coordinates" not in annotation["geometry"]) or ("classification" not in annotation["properties"]):
            format_okay = False
        elif ("name" not in annotation["properties"]["classification"]) or ("color" not in annotation["properties"]["classification"]):
            format_okay = False
        if not format_okay:
            print("Improperly formatted annotation - skipping...")
            continue
        geo_type = annotation["geometry"]["type"]
        if geo_type == 'Polygon':
            coordinates = annotation["geometry"]["coordinates"][0]
            shape = Polygon(coordinates)
        elif geo_type == 'MultiPolygon':
            # Typically one big polygon and a few little ones; use just the big one
            multi_coordinates = annotation["geometry"]["coordinates"]
            lengths = [len(x[0]) for x in multi_coordinates]
            index = lengths.index(max(lengths))
            coordinates = multi_coordinates[index][0]
            shape = Polygon(coordinates)
        elif geo_type == 'LineString':
            coordinates = annotation["geometry"]["coordinates"]
            shape = LineString(coordinates)
        else:
            print("Unknown geometry type: " + geo_type)
            continue
        class_name = annotation["properties"]["classification"]["name"]
        class_name = class_name.replace(' ','_')
        class_name = class_name.replace('*','')
        class_name = class_name.replace('/','')
        class_name = class_name.replace("'",'')
        class_name = class_name.replace(".",'')
        class_name = class_name.lower()
        color = annotation["properties"]["classification"]["color"] # [red, green, blue]
        shapes.append(shape)
        class_names.append(class_name)
        colors.append(color)
    return shapes, class_names, colors

def save_colors(tiles_dir, class_names, colors):
    color_file = os.path.join(tiles_dir, 'colors.json')
    if os.path.exists(color_file):
        with open(color_file, "r") as f:
            color_map = json.load(f)
    else:
        color_map = {}
    for class_name, color in zip(class_names, colors):
        if class_name not in color_map.keys():
            color_map.update({class_name: color})
    with open(color_file, "w") as f:
        json.dump(color_map, f)

def intersects_enough(tile_polygon, shape, min_area):
    """Returns True if tile_polygon intersects shape by at least min_area
    or the tile encompasses the shape. Note that shape may not be a Polygon
    or LineString due to the Shapely make_valid call in generate_tiles()."""
    if shape.within(tile_polygon):
        return True
    intersection = tile_polygon.intersection(shape)
    if isinstance(shape, LineString):
        intersecting_points = list(intersection.coords)
        if len(intersecting_points) > 0:
            return True # Comment out to use version below
            # The version below respects min_overlap
            area1, area2 = line_polygon_areas(tile_polygon, intersecting_points)
            area = min(area1, area2)
            if (area >= min_area):
                return True
    else:
        area = intersection.area
        if (area >= min_area):
            return True
    return False

def line_polygon_areas(polygon, intersecting_points):
    """Returns the areas of the two polygons defined by the line through the
    given intersecting points and the polygon (Shapely Polygon). Approximates
    the two polygons as convex hulls.
    NOTE: Code currently not used. See comments in intersects_enough()."""
    area1 = area2 = 0.0
    if len(intersecting_points) > 1:
        # Represent intersection as straight line between intersecting points 
        (x1,y1) = point1 = intersecting_points[0]
        (x2,y2) = point2 = intersecting_points[-1]
        # Separate polygon points into left and right of line
        left_points = []
        right_points = []
        poly_points = list(polygon.exterior.coords)
        for point3 in poly_points:
            (x3,y3) = point3
            cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross_product > 0:
                left_points.append(point3)
            else:
                right_points.append(point3)
        if len(left_points) > 0:
            points = left_points + [point1, point2]
            hull = ConvexHull(points)
            area1 = hull.volume
        if len(right_points) > 0:
            points = right_points + [point1, point2]
            hull = ConvexHull(points)
            area2 = hull.volume
    return area1, area2

def compute_tiles(args, image, shape):
    """Compute and return tiles (x,y,w,h) for all tiles in image overlapping shape."""
    tile_size = args.tile_size
    tile_increment = args.tile_increment
    tile_overlap = args.tile_overlap
    min_area = tile_size * tile_size * tile_overlap
    if isinstance(shape, LineString):
        min_area = min_area / 2
    if tile_increment is None:
        tile_increment = tile_size
    width, height = image.dimensions
    (minx,miny,maxx,maxy) = shape.bounds
    # Only consider tiles around and inside shape
    xmin = max(0, (int(minx) - tile_size))
    ymin = max(0, (int(miny) - tile_size))
    xmax = min(width, (int(maxx) + tile_size))
    ymax = min(height, (int(maxy) + tile_size))
    x1 = xmin
    y1 = ymin
    x2 = x1 + tile_size
    y2 = y1 + tile_size
    tiles = []
    while y2 <= ymax:
        while x2 <= xmax:
            tile_polygon = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1)])
            if intersects_enough(tile_polygon, shape, min_area):
                tiles.append([x1, y1, tile_size, tile_size])
            x1 += tile_increment
            x2 += tile_increment
        x1 = xmin
        x2 = x1 + tile_size
        y1 += tile_increment
        y2 += tile_increment
    return tiles

def generate_tiles(args, image):
    num_tiles = 0
    shapes, class_names, colors = parse_qupath_annotations(args.annotations_file)
    save_colors(args.tiles_dir, class_names, colors)
    unique_class_names = list(set(class_names))
    for class_name in unique_class_names:
        class_name_dir = os.path.join(args.tiles_dir, 'images', class_name)
        os.makedirs(class_name_dir, exist_ok=True)
    print('Generating tiles...', flush=True)
    for shape, class_name in zip(shapes, class_names):
        if isinstance(shape, Polygon):
            shape = make_valid(shape) # fix self-intersecting areas
            # Note: call to make_valid may not return Polygon
        tiles = compute_tiles(args, image, shape)
        for tile in tiles:
            num_tiles += 1
            write_tile(args, image, num_tiles, tile, class_name)
    return num_tiles

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_file', type=str, required=True) 
    parser.add_argument('--annotations', dest='annotations_file', type=str, required=True)
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str, required=True)
    parser.add_argument('--tile_size', dest='tile_size', type=int, default=300)
    parser.add_argument('--tile_increment', dest='tile_increment', type=int, default=None)
    parser.add_argument('--tile_overlap', dest='tile_overlap', type=float, default=0.9)
    parser.add_argument('--enhance', action='store_true')
    args = parser.parse_args()
    return args

def main():
    # Read arguments
    args = parse_arguments()
    image_file = args.image_file
    tiles_dir = args.tiles_dir
    init_files(tiles_dir)
    # Read SVS image
    print("Reading image \"" + image_file + "\"...", flush=True)
    image = OpenSlide(image_file)
    # Generate tiles
    num_tiles = generate_tiles(args, image)
    print("Generated " + str(num_tiles) + " tile images")
    return

if __name__ == '__main__':
    main()
