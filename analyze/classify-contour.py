# classify-contour.py --contour_file <contour_file> --tiles_dir <tiles_dir>
#
# Reads in contour prediction information from <tiles_dir>/tiles.csv file
# and computes the distribution of obex vs. node tiles for each contour.
# The max of the distribution is written back to the contour file in the
# classification name property.
#
# Also reads in stained_pixels information from the tiles.csv file and
# adds to the contour file properties for the #stained_pixels and
# #total_pixels.
#
# Author: Lawrence Holder, Washington State University

import matplotlib.pyplot as plt
import json
import os
import argparse
import csv
import numpy as np

TISSUE_TYPE_PREDICTION_COLUMN = 'tissue_type_prediction'

def get_max_key(distribution):
    max_key = ''
    max_value = -1
    for key,value in distribution.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key

def plot_distribution(contour_file, distribution):
    """Write plot image showing distribution, which is dictionary of type:count objects."""
    contour_file_noext = os.path.splitext(contour_file)[0]
    plot_file_name = contour_file_noext + '-distribution.png'
    max_key = get_max_key(distribution)
    certainty = compute_certainty(distribution)
    title = f'{max_key} (certainty={certainty:.6f})'
    outcomes = sorted(distribution.keys())
    counts = [distribution[key] for key in outcomes]
    # Creating the bar chart
    plt.figure(figsize=(8, 4))  # Set the figure size (optional)
    plt.bar(outcomes, counts, color=['blue', 'green'])  # Set colors for each bar (optional)
    # Adding the text labels on the bars
    for i in range(len(counts)):
        plt.text(i, counts[i] + 0.01, f'{counts[i]}', ha='center', va='bottom', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=14, fontweight='bold')
    if len(counts) > 0:
        plt.ylim(0, max(counts) * 1.1)  # Extend y-axis to make room for text
    # Save the plot
    plt.savefig(plot_file_name, dpi=300)
    #plt.show()
    plt.close()
    return

def print_distribution(contour_file, distribution):
    contour_file_base = os.path.basename(contour_file)
    contour_name = os.path.splitext(contour_file_base)[0]
    #max_key = get_max_key(distribution)
    #print(f"{contour_name}: {str(max_key)}")
    certainty = compute_certainty(distribution)
    print(f"{contour_name}: {str(distribution)} (certainty={certainty})")
    return

def process_predictions(csv_file):
    """Compute distribution over predictions present in tiles file.
    Also sum the number of stained and total pixels."""
    stained_pixels = 0
    total_pixels = 0
    predictions_dict = {}
    with open(csv_file, 'r') as csv_file:
        csv_dict_reader = csv.DictReader(csv_file)
        for row in csv_dict_reader:
            # Collect prediction info
            prediction = row[TISSUE_TYPE_PREDICTION_COLUMN]
            if prediction not in predictions_dict:
                predictions_dict[prediction] = 0
            predictions_dict[prediction] += 1
            # Collect stained pixel info
            w = int(row['width'])
            h = int(row['height'])
            sp = int(row['stained_pixels'])
            stained_pixels += sp
            total_pixels += (w * h)
    return predictions_dict, stained_pixels, total_pixels

def compute_certainty(distribution):
    counts = [distribution[tissue_type] for tissue_type in distribution.keys()]
    total = sum(counts)
    probs = np.array([(count / total) for count in counts if count > 0])
    entropy = -np.sum(probs * np.log2(probs))
    return (1.0 - entropy)

def add_classification(contour_file, distribution, stained_pixels, total_pixels):
    max_key = get_max_key(distribution)
    certainty = compute_certainty(distribution)
    new_features = []
    with open(contour_file, 'r') as cf:
        features = json.load(cf)
        for feature in features:
            geometry = feature['geometry']
            if geometry['type'].lower() == 'polygon':
                properties = feature['properties']
                classification = properties['classification']
                classification['name'] = max_key
                classification['certainty'] = certainty
                properties['stained_pixels'] = stained_pixels
                properties['total_pixels'] = total_pixels
                new_features.append(feature)
    with open(contour_file, 'w') as cf:
        json.dump(new_features, cf, indent=2)
    return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contour_file', dest='contour_file', type=str, required=True)
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    contour_file = args.contour_file
    tiles_csv_file = os.path.join(args.tiles_dir, 'tiles.csv')
    distribution, stained_pixels, total_pixels = process_predictions(tiles_csv_file)
    print_distribution(contour_file, distribution)
    plot_distribution(contour_file, distribution)
    add_classification(contour_file, distribution, stained_pixels, total_pixels)
    return

if __name__ == "__main__":
   main()
