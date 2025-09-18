# annotate.py --image <svs_slide> --geojson <geojson_file>
#   --errors <color> --class <class> --probability <prob> --tiles_dir <tiles_dir> 
#
# General tool for scaling and annotating a large SVS slide image. If a tiles_dir
# is given, then tiles stored under the same slide name in the tiles.csv file are
# written to the image. If the --errors option is given, then the tiles directory
# is assumed to have a tiles_predicted.csv file, and tiles with incorrect
# predictions are annotated in the given color. If the --class option is given,
# then the tiles directory is assumed to have a tiles_predicted.csv file, and only
# tiles with the predicted class are annotated in the color given in the colors.json
# file. If --probability is given, then only tiles whose predicted class has a
# probability >= <prob> are annotated.
#
# If a geojson_file is given, then the shapes from the file are written to the image.
#
# All coordinates are assumed to be in the slide's original dimensions. The result
# is written to an image file with same name, plus '-annotated.png'. If --errors is
# given, then errant tiles are also output as a GeoJSON file for import into QuPath.
#
# TODO: Make sure all options are consistent with each other and/or prevent illegal
# combinations.
#
# Author: Lawrence Holder, Washington State University

import os
import argparse
import json
import csv
from PIL import ImageDraw, ImageColor
from openslide import OpenSlide
import geojson

SCALE_FACTOR = 32

def scale_image(slide):
    w, h = slide.dimensions
    new_size = (w / SCALE_FACTOR, h / SCALE_FACTOR)
    img = slide.get_thumbnail(size=new_size)
    return img

def add_tiles(image, image_base_name, tiles_dir):
    """Add tiles in <tiles_dir>/tiles.csv for entries matching <image_base_name>.
    Returns tiles as [x,y,width,height,color,name] for later output as GeoJSON."""
    # Read colors
    color_file_name = os.path.join(tiles_dir, 'colors.json')
    with open(color_file_name, 'r') as color_file:
        color_dict = json.load(color_file)
    # Read tiles and write those for image name
    tiles_file_name = os.path.join(tiles_dir, 'tiles.csv')
    image_base_name = os.path.basename(image_base_name) # remove path
    draw = ImageDraw.Draw(image)
    tiles = []
    with open(tiles_file_name, 'r') as tiles_file:
        for row in csv.DictReader(tiles_file):
            # Remove image number from image name and compare
            image_name_split = row['image'].split('_')
            image_name = '_'.join(image_name_split[:-1])
            if True: #image_name == image_base_name:
                x = int(row['x'])
                y = int(row['y'])
                w = int(row['width'])
                h = int(row['height'])
                #feature = row['feature']
                feature = row['directory']
                if 'idline' not in feature:
                    color = tuple(color_dict[feature])
                    tile = [x, y, w, h, color, feature + '_tile']
                    # Scale tile location and size
                    x = x // SCALE_FACTOR
                    y = y // SCALE_FACTOR
                    w = w // SCALE_FACTOR
                    h = h // SCALE_FACTOR
                    # Add to image
                    rect_coords = (x, y, x+w, y+h)
                    draw.rectangle(rect_coords, outline=color, width=1)
                    tiles.append(tile)
    return tiles

def add_tile_errors(image, image_base_name, tiles_dir, error_color):
    """Add tiles in <tiles_dir>/tiles_predicted.csv for entries matching <image_base_name>,
    and when the feature and predicted labels do not match. Color these errant tiles in the
    given color. Returns tiles as [x,y,width,height,color,name] for later output as GeoJSON."""
    # Read colors
    color_file_name = os.path.join(tiles_dir, 'colors.json')
    with open(color_file_name, 'r') as color_file:
        color_dict = json.load(color_file)
    # Read tiles and draw on image
    tiles_file_name = os.path.join(tiles_dir, 'tiles_predicted.csv')
    image_base_name = os.path.basename(image_base_name) # remove path
    draw = ImageDraw.Draw(image)
    tiles = []
    with open(tiles_file_name, 'r') as tiles_file:
        for row in csv.DictReader(tiles_file):
            # Remove image number from image name and compare
            image_name_split = row['image'].split('_')
            image_name = '_'.join(image_name_split[:-1])
            if image_name == image_base_name:
                x = int(row['x'])
                y = int(row['y'])
                w = int(row['width'])
                h = int(row['height'])
                feature = row['feature']
                color = tuple(color_dict[feature])
                #tile = [x, y, w, h, color, feature + '_tile']
                if feature != row['predicted']:
                    tile = [x, y, w, h, error_color, 'error_tile']
                    tiles.append(tile)
                    # Scale tile location and size
                    x = x // SCALE_FACTOR
                    y = y // SCALE_FACTOR
                    w = w // SCALE_FACTOR
                    h = h // SCALE_FACTOR
                    # Add to image
                    rect_coords = (x, y, x+w, y+h)
                    draw.rectangle(rect_coords, outline=error_color, width=1)
    return tiles

def add_tile_predictions(image, image_base_name, tiles_dir, class_name, prob_threshold):
    """Add tiles in <tiles_dir>/tiles_predicted.csv for entries matching <image_base_name>,
    and when the predicted label matches the given class, and the prediction probability is
    at least the given prob_threshold. Returns tiles as [x,y,width,height,color,name] for
    later output as GeoJSON."""
    # Read colors
    color_file_name = os.path.join(tiles_dir, 'colors.json')
    with open(color_file_name, 'r') as color_file:
        color_dict = json.load(color_file)
    color = tuple(color_dict[class_name])
    # Read tiles and draw on image
    tiles_file_name = os.path.join(tiles_dir, 'tiles_predicted.csv')
    image_base_name = os.path.basename(image_base_name) # remove path
    draw = ImageDraw.Draw(image)
    tiles = []
    with open(tiles_file_name, 'r') as tiles_file:
        for row in csv.DictReader(tiles_file):
            # Remove image number from image name and compare
            image_name_split = row['image'].split('_')
            image_name = '_'.join(image_name_split[:-1])
            if image_name == image_base_name:
                x = int(row['x'])
                y = int(row['y'])
                w = int(row['width'])
                h = int(row['height'])
                predicted_class = row['predicted']
                probability = float(row['probability'])
                if (predicted_class == class_name) and (probability >= prob_threshold):
                    tile = [x, y, w, h, color, predicted_class + '_tile']
                    tiles.append(tile)
                    # Scale tile location and size
                    x = x // SCALE_FACTOR
                    y = y // SCALE_FACTOR
                    w = w // SCALE_FACTOR
                    h = h // SCALE_FACTOR
                    # Add to image
                    rect_coords = (x, y, x+w, y+h)
                    draw.rectangle(rect_coords, outline=color, width=1)
    return tiles

def add_geojson(image, geojson_file_name):
    draw = ImageDraw.Draw(image)
    with open(geojson_file_name, 'r') as geof:
        features = json.load(geof)
        for feature in features:
            geometry = feature['geometry']
            properties = feature['properties']
            classification = properties['classification']
            color = tuple(classification['color'])
            geo_type = geometry["type"]
            if geo_type == 'Polygon':
                coordinates = geometry["coordinates"][0]
            elif geo_type == 'MultiPolygon':
                # Typically one big polygon and a few little ones; use just the big one
                multi_coordinates = geometry["coordinates"]
                lengths = [len(x[0]) for x in multi_coordinates]
                index = lengths.index(max(lengths))
                coordinates = multi_coordinates[index][0]
            elif geo_type == 'LineString':
                coordinates = geometry["coordinates"]
            points = []
            for point in coordinates:
                x = point[0] // SCALE_FACTOR
                y = point[1] // SCALE_FACTOR
                points.append((x,y))
            if geo_type == 'LineString':
                draw.line(points, fill=color, width=2)
            else:
                draw.polygon(points, outline=color, width=2)
    return

def write_tiles_as_geojson(image_file_name, tiles):
    """Writes tiles, each formatted as [x,y,w,h,color], as GeoJSON polygons."""
    feature_arr = []
    for tile in tiles:
        x,y,w,h,color,name = tile
        properties = {
            "objectType": "annotation",
            "classification": {
                "name": str(name),
                "color": list(color)
            }
        }
        tile_coords = [[x,y], [x+w,y], [x+w,y+h], [x,y+h], [x,y]]
        feature = geojson.Feature(geometry=geojson.Polygon([tile_coords]), properties=properties)
        feature_arr.append(feature)
    with open(image_file_name + '.tile_annotations.json', 'w') as geojson_file:
        geojson.dump(feature_arr, geojson_file, indent=2)
    return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_file', type=str, required=True) 
    parser.add_argument('--geojson', dest='geojson_file', type=str)
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str)
    parser.add_argument('--errors', dest='error_color', type=str)
    parser.add_argument('--class', dest='class_name', type=str)
    parser.add_argument('--probability', dest='probability', default=0.0, type=float)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    image_file_name = args.image_file
    geojson_file_name = args.geojson_file
    tiles_dir = args.tiles_dir
    error_color = args.error_color
    class_name = args.class_name
    prob_threshold = args.probability
    if error_color:
        error_color_rgb = ImageColor.getcolor(args.error_color, "RGB")
    image_base_name, _ = os.path.splitext(image_file_name)
    # Read SVS image
    print("Reading image \"" + image_file_name + "\"...", flush=True)
    slide = OpenSlide(image_file_name)
    # Scale image
    print('Scaling image...')
    scaled_image = scale_image(slide)
    # Add geojson shapes (if given)
    if geojson_file_name:
        print('Adding geojson annotations')
        add_geojson(scaled_image, geojson_file_name)
    # Add tiles (if given)
    if tiles_dir:
        print('Adding tile annotations')
        if class_name:
            tiles = add_tile_predictions(scaled_image, image_base_name, tiles_dir, class_name, prob_threshold)
        else:
            tiles = add_tiles(scaled_image, image_base_name, tiles_dir)
        if error_color:
            tiles = add_tile_errors(scaled_image, image_base_name, tiles_dir, error_color_rgb)
        write_tiles_as_geojson(image_file_name, tiles)
    # Write image
    outfile = image_base_name + '-annotated.png'
    scaled_image.save(outfile)
    return

if __name__ == "__main__":
    main()
