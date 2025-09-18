# add-classification.py --contour <contour_file> --class <class> [--color <color>]
#
# Sets the classification name of each contour in the contour_file to class.
# If color is given, then each classification color is set to the color.
#
# Author: Lawrence Holder, Washington State University

import sys
import json
import argparse
from PIL import ImageColor

def color_name_to_rgb(color_name):
    try:
        rgb = ImageColor.getrgb(color_name)
        return rgb
    except ValueError:
        return None

def process_contours(contour_file_name, class_name, color_name):
    color = None
    if color_name:
        color = color_name_to_rgb(color_name)
        if not color:
            print(f"Error: unknown color \"{color_name}\"")
            sys.exit()
    with open(contour_file_name, 'r') as contour_file:
        features = json.load(contour_file)
        new_features = []
        for feature in features:
            properties = feature['properties']
            classification = properties['classification']
            classification['name'] = class_name
            if color:
                classification['color'] = list(color)
            new_features.append(feature)
    with open(contour_file_name, 'w') as contour_file:
        json.dump(new_features, contour_file, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contour', dest='contour_file_name', type=str, required=True)
    parser.add_argument('--class', dest='class_name', type=str, required=True)
    parser.add_argument('--color', dest='color_name', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    process_contours(args.contour_file_name, args.class_name, args.color_name)
    return

if __name__ == "__main__":
    main()
