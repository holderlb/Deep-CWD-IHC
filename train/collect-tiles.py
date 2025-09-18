# collect-tiles.py --tiles_dir <tile_dir> --train_dir <train_dir>
#   --class_names class_names --class_column <class_column>
#   --sample_rate <N.N>
#
# Searches in tiles_dir for tiles with one of the given class names in the
# given class_column and copies the images to train_dir/class_name in
# preparation for training. Creates destination directories if they do not
# already exist.
#
# Each class of tiles is independently sampled at the given sample_rate.
# Default sample_rate is 1.0.
#
# Author: Lawrence Holder, Washington State University

import argparse
import os
import csv
import shutil
import random
import sys

CSV_CLASS_NAME_COLUMN = 'directory'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str, required=True)
    parser.add_argument('--train_dir', dest='train_dir', type=str, required=True)
    parser.add_argument('--class_names', nargs='+', dest='class_names', type=str, required=True)
    parser.add_argument('--class_column', type=str, default=CSV_CLASS_NAME_COLUMN)
    parser.add_argument('--sample_rate', dest='sample_rate', type=float, default=1.0)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    tiles_dir = args.tiles_dir
    train_dir = args.train_dir
    class_names = args.class_names
    class_column = args.class_column
    sample_rate = args.sample_rate
    if sample_rate > 1.0:
        print('collect-tiles error: sample rate cannot exceed 1.0')
        sys.exit()
    class_lists = {}
    # Ensure training directories exist
    os.makedirs(train_dir, exist_ok=True)
    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        class_lists[class_name] = []
    # Open tiles_dir/tiles.csv and collect tile image names from class names
    tiles_file_name = os.path.join(tiles_dir, 'tiles.csv')
    with open(tiles_file_name, 'r') as tiles_file:
        csv_reader = csv.DictReader(tiles_file)
        for row in csv_reader:
            class_name = row[class_column]
            if class_name in class_names:
                class_lists[class_name].append(row['image'])
    for class_name in class_names:
        init_count = len(class_lists[class_name])
        sample_count = int(init_count * sample_rate)
        print(f'  Class: {class_name}: sampled {sample_count} of {init_count} tiles')
        sample_list = random.sample(class_lists[class_name], sample_count)
        for image_name in sample_list:
            image_file_name = image_name + '.png'
            source = os.path.join(tiles_dir, 'images', class_name, image_file_name)
            destination = os.path.join(train_dir, class_name)
            shutil.copy2(source, destination)
    return

if __name__ == '__main__':
    main()