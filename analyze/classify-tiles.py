# classify-tiles.py  --model <model_file> --tiles_dir <tiles_dir> --class <class_name>
#
# Classifies the tiles in <tiles_dir> according to the model in <model_file>.
# New columns called <class_name>_prediction and <class_name>_probability are
# added to the 'tiles.csv'.
#
# The <tiles_dir> is assumed to contain a 'tiles.csv' file that describes each
# tile and an 'images' directory that contains the actual tile images. The images
# are in subdirectories according to the 'class' column value in the 'tiles.csv'
# file.
#
# The model is stored in two files: <model_file>.keras and <model_file>.json.
# The JSON file contains the class values predicted by the model.
#
# Author: Lawrence Holder, Washington State University

import os
import numpy as np
import argparse
import json
import pandas as pd
import tensorflow as tf
from tensorflow import keras

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
NUM_IMAGES_PER_TEST = 100
TILE_CSV_FILE = 'tiles.csv'

def load_test_images(tiles_df, tile_dir, start_row, end_row):
    """Load tile images in tiles_df[start_row,end_row] from tile_dir/images."""
    images = []
    for row in tiles_df.iloc[start_row:end_row].itertuples(index=False):
        image_name = row.image
        directory_name = row.directory
        image_file = os.path.join(tile_dir, 'images', directory_name, image_name + '.png')
        image = read_image(image_file)
        images.append(image)
    return images

def read_image(image_path):
    """Read image from file, ignoring alpha channel, resizing if necessary. Return as NumPy array."""
    img = keras.preprocessing.image.load_img(image_path, color_mode='rgb', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    img = keras.preprocessing.image.img_to_array(img)
    return img

def test_model(model, tiles_df, tiles_dir, class_names):
    """Returns class labels, and their probability, predicted by given model
    on given tiles. Done incrementally to reduce memory use."""
    n = NUM_IMAGES_PER_TEST
    labels = []
    probs = []
    num_rows = len(tiles_df)
    for i in range(0, num_rows, n):
        #print(f"Testing images {i} to {i+n-1} of {num_rows}")
        img_subset = load_test_images(tiles_df, tiles_dir, i, i+n)
        pred_subset = model.predict(np.array(img_subset), verbose=0)
        label_subset = [class_names[list(pred).index(max(pred))] for pred in pred_subset]
        prob_subset = [max(pred) for pred in pred_subset]
        labels.extend(label_subset)
        probs.extend(prob_subset)
    return labels, probs

def read_model(model_file):
    print(f'Reading model: {model_file}')
    model = tf.keras.models.load_model(model_file + '.keras')
    model_json_file = model_file + '.json'
    with open(model_json_file, 'r') as mj_file:
        model_dict = json.load(mj_file)
    class_names = model_dict['class_names']
    return model, class_names

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class', dest='class_name', type=str) 
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str)
    parser.add_argument('--model', dest='model_file', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    class_name = args.class_name
    tiles_dir = args.tiles_dir
    tiles_csv_file = os.path.join(tiles_dir, TILE_CSV_FILE)
    tiles_df = pd.read_csv(tiles_csv_file)
    model, class_values = read_model(args.model_file)
    labels, probs = test_model(model, tiles_df, tiles_dir, class_values)
    tiles_df[class_name + '_prediction'] = labels
    tiles_df[class_name + '_probability'] = probs
    tiles_df.to_csv(tiles_csv_file, index=False)
    return

if __name__ == "__main__":
   main()
