# train.py --image_file <image_file> --contour_class_file <contour_class_file>
#   --train_dir <train_dir> --sample_rate <N.N> --enhance --background
#
# Top level script to train models on annotations of given image_file, assumed
# to be in SVS format. When generating tissue tiles, the contour_class_file is
# expected, a CSV file with the image, contour number, and tissue type. When
# collecting tiles, the tiles are copied to the given train_dir. Each tile
# class is sampled at the given sample_rate (default=1.0). If --enhance is
# given, then contrast is enhanced for all training images and tiles. If
# --background is given, then tasks are run in the background (CAREFUL:
# calling train multiple times with --background can saturate machine).
#
# Author: Lawrence Holder, Washington State University

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
import argparse
import csv

MODEL_DIR = '../models'
TISSUE_MODEL = 'model-tissue'
NODE_MODEL = 'model-node'
OBEX_MODEL = 'model-obex'
CONTOUR_CLASSES_FILE = 'contour_classes.csv'
TILE_SIZE = 300
TILE_OVERLAP = 0.9
TILE_INCREMENT = 300

def filter_contour_files(image_file):
    """Returns contour JSON files matching image file."""
    image_file_path = os.path.dirname(image_file)
    image_file_base = os.path.basename(image_file)
    image_file_base_noext = os.path.splitext(image_file_base)[0]
    directory_path = Path(image_file_path)
    pattern = image_file_base_noext + '-contour-*.json'
    file_names = list(directory_path.glob(pattern))
    contour_file_names = []
    for file_name in file_names:
        file_name_base = os.path.basename(file_name)
        file_name_ext = os.path.splitext(file_name_base)[1]
        components = file_name_base.split('-')
        if (file_name_ext == '.json') and (len(components) > 1):
            if components[-2] == 'contour':
                contour_file_names.append(file_name_base)
    contour_file_names.sort()
    return contour_file_names


def segment_image(image_file, enhance=False):
    """Find contours in image."""
    command = ['python', '../common/segment.py']
    command += ['--image', image_file]
    if enhance:
        command += ['--enhance']
    subprocess.run(command)
    return

def generate_annotations_training_tiles(image_file, enhance=False, background=False):
    """Generate tiles for image according to image annotations file."""
    annotations_file_name = image_file + '.annotations.json'
    tiles_dir = image_file + '.annotations-training-tiles'
    command = ['python', '../common/generate-tiles.py']
    command += ['--image', image_file, '--tiles_dir', tiles_dir]
    command += ['--annotations', annotations_file_name]
    command += ['--tile_size', str(TILE_SIZE), '--tile_overlap', str(TILE_OVERLAP)]
    command += ['--tile_increment', str(TILE_INCREMENT)]
    if enhance:
        command += ['--enhance']
    if background:
        subprocess.Popen(command)
    else:
        subprocess.run(command)
    return

def add_contour_classifications(image_file, contour_class_file):
    num_contours = 0
    with open(contour_class_file, 'r') as contour_csv_file:
        csv_dict_reader = csv.DictReader(contour_csv_file)
        image_file_base = os.path.basename(image_file)
        for row in csv_dict_reader:
            if image_file_base == row['image']:
                image_file_root, _ = os.path.splitext(image_file)
                contour_file = image_file_root + "-contour-" + str(row['contour']) + ".json"
                print(f'  Processing contour file {contour_file}')
                tissue_type = row['class']
                color_name = "green" # obex
                if tissue_type == "node":
                    color_name = "blue"
                command = ["python", "add-classification.py"]
                command += ["--contour", contour_file, "--class", tissue_type, "--color", color_name]
                subprocess.run(command)
                num_contours += 1
    print(f'  Added classification to {num_contours} contour files')
    return

def generate_tissue_training_tiles(image_file, enhance=False, background=False):
    """Generate tiles for each contour of image according to its tissue classification."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        command = ['python', '../common/generate-tiles.py']
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-training-tiles')
        command += ['--image', image_file, '--tiles_dir', tiles_dir]
        contour_file = os.path.join(image_file_path, contour_file_name)
        print(f'  Processing contour file {contour_file}', flush=True)
        command += ['--annotations', contour_file]
        command += ['--tile_size', str(TILE_SIZE), '--tile_overlap', str(TILE_OVERLAP)]
        if enhance:
            command += ['--enhance']
        if background:
            subprocess.Popen(command)
        else:
            subprocess.run(command)
    return

def collect_tissue_training_tiles(image_file, train_dir, sample_rate):
    """Collect sample of tissue tiles for each contour of image."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        contour_file = os.path.join(image_file_path, contour_file_name)
        print(f'  Processing contour file {contour_file}', flush=True)
        command = ['python', 'collect-tiles.py']
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-training-tiles')
        command += ['--tiles_dir', tiles_dir, '--train_dir', train_dir]
        command += ['--class_names', 'node', 'obex']
        command += ['--class_column', 'directory']
        command += ['--sample_rate', str(sample_rate)]
        subprocess.run(command)
    return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', dest='image_file', type=str, required=True)
    parser.add_argument('--contour_class_file', dest='contour_class_file', type=str)
    parser.add_argument('--train_dir', dest='train_dir', type=str)
    parser.add_argument('--sample_rate', dest='sample_rate', type=float, default=1.0)
    parser.add_argument('--enhance', action='store_true')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    image_file = args.image_file
    enhance = args.enhance
    background = args.background
    
    # Generate tiles based on the manual annotations extracted from QuPath
    generate_annotations_training_tiles(image_file, enhance, background)

    # Collect tiles for dorsal_motor_nucleus/not_dmn (optionally sample)
    # Train obex model for dorsal_motor_nucleus vs. not_dmn
    # Collect tiles for follicle/non_follicular (optionally sample)
    # Train node model for follicle vs. non_follicular

    # Generate tiles based on the tissue type of each contour found by segment.py.
    # Assumes the presence of a contours.csv file that has a row for each contour:
    # image name, contour #, tissue type (node or obex).
    segment_image(image_file, enhance)
    add_contour_classifications(image_file, args.contour_class_file)
    
    # Generate tissue tiles for each contour
    generate_tissue_training_tiles(image_file, enhance, background)

    # Collect tiles for node/obex (optionally sample)
    collect_tissue_training_tiles(image_file, args.train_dir, args.sample_rate)

    # Train tissue model for node vs. obex

    return

if __name__ == "__main__":
    main()
