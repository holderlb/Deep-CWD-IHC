# analyze.py --image_file <image_file> --enhance --background
#            [--metadata <metadata_file>]
#
# Top level script to analyze image. Assumes SVS image file with .svs extension.
# If --enhance is given, then image contrast is enhanced. If --background is
# given, then tile generation processes are spawned in the background in
# parallels.
#
# If metadata file is given, this is assumed to be a JSON-formatted file containing
# parameter values related to DMN and follicle properties. See "Metadata keys" section
# below.
#
# Author: Lawrence Holder, Washington State University

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
import argparse

MODEL_DIR = '../models'
TISSUE_MODEL = 'model-tissue'
NODE_MODEL = 'model-node'
OBEX_MODEL = 'model-obex'
MIDLINE_MODEL = 'model-midline'
TILE_SIZE = 300
TILE_OVERLAP = 1.0 # Change this to be consistent with overlap used for model training

# Metadata keys
MINIMUM_PROBABILITY_FOLLICLE_TILE_KEY = 'minimum_probability_follicle_tile'
MINIMUM_AREA_FOLLICLE_REGION_KEY = 'minimum_area_follicle_region'
MINIMUM_PROBABILITY_DMN_TILE_KEY = 'minimum_probability_dmn_tile'
MINIMUM_AREA_DMN_REGION_KEY = 'minimum_area_dmn_region'
MINIMUM_PERCENT_STAINED_PIXELS_NODE_KEY = 'minimum_percent_stained_pixels_node'
MINIMUM_PERCENT_STAINED_PIXELS_OBEX_KEY = 'minimum_percent_stained_pixels_obex'
MINIMUM_NUMBER_FOLLICLES_KEY = 'minimum_number_follicles'
SHOW_DETAILS_FLAG_KEY = 'show_details_flag'

# Subset of region detection methods that will be included in webpage (in the order given)
# See detect-regions.py for possible values. Default method used for basic results output.
REGION_DETECTION_METHODS = ['diagonal', 'cellpose', 'adjacent4', 'watershed', 'graph', 'gaussian', 'adjacent2', 'adjacent1']
REGION_DETECTION_METHODS = ['diagonal', 'graph_0.5', 'graph_0.6', 'graph_0.7', 'graph_0.8', 'graph_0.89',
                            'gaussian_10', 'gaussian_20', 'gaussian_30', 'gaussian_40', 'gaussian_50']
REGION_DETECTION_METHODS = ['diagonal_1.0', 'diagonal_0.50', 'diagonal_0.25'] # min area threshold reduced by given argument
REGION_DETECTION_METHODS = ['diagonal'] # 'diagonal' is the default (best?) method
DEFAULT_DETECTION_METHOD = 'diagonal'

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

def generate_tiles(image_file, enhance=False, background=False):
    """Generate tiles for each contour."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    processes = []
    for contour_file_name in contour_file_names:
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        print(f'  Generating tiles for contour {contour_file_name}')
        command = ['python', '../common/generate-tiles.py']
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
        command += ['--image', image_file, '--tiles_dir', tiles_dir]
        contour_file = os.path.join(image_file_path, contour_file_name)
        command += ['--annotations', contour_file]
        command += ['--tile_size', str(TILE_SIZE), '--tile_overlap', str(TILE_OVERLAP)]
        if enhance:
            command += ['--enhance']
        if background:
            processes.append(subprocess.Popen(command))
        else:
            subprocess.run(command)
    if background:
        # Block until all processes finished
        for proc in processes:
            proc.wait()
    return

def classify_tiles_tissue(image_file):
    """Classify tiles in each contour as node or obex"""
    print('  Classifying tile tissue type...')
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        print(f'  Classifying tissue type for tiles in contour {contour_file_name}')
        command = ['python', 'classify-tiles.py']
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
        command += ['--class', 'tissue_type', '--tiles_dir', tiles_dir]
        model_file = os.path.join(MODEL_DIR, TISSUE_MODEL)
        command += ['--model', model_file]
        subprocess.run(command)
    return

def classify_contours(image_file):
    """Classify contour as node or obex based on majority of tile classifications.
    Write result to contour json file and generate distribution plot."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        print(f'  Classifying tissue type for contour {contour_file_name}')
        command = ['python', 'classify-contour.py']
        contour_file = os.path.join(image_file_path, contour_file_name)
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
        command += ['--contour_file', contour_file, '--tiles_dir', tiles_dir]
        subprocess.run(command)
    return

def get_tissue_type(contour_file):
    """Returns tissue type classification stored in the last polygon feature of contour file.
    Should be only one feature in the file, but if more than one, they are likely to
    all have the same tissue type."""
    tissue_type = None
    with open(contour_file, 'r') as cf:
        features = json.load(cf)
        for feature in features:
            if feature['geometry']['type'].lower() == 'polygon':
                properties = feature['properties']
                classification = properties['classification']
                tissue_type = classification['name']
                certainty = classification['certainty']
    # Choose node tissue by default
    if (tissue_type == None) or (tissue_type == ""):
        tissue_type = 'node'
    return tissue_type.lower(), certainty

def get_contrast(contour_file):
    """Returns contrast information stored in the last polygon feature of contour file.
    Should be only one feature in the file, but if more than one, they are likely to
    all have the same contrast. Values are 'low' or 'normal'."""
    contrast = 'normal'
    with open(contour_file, 'r') as cf:
        features = json.load(cf)
        for feature in features:
            if feature['geometry']['type'].lower() == 'polygon':
                properties = feature['properties']
                contrast = properties['contrast']
    return contrast

def get_stained_pixels(contour_file):
    """Returns stained pixel information stored in the last polygon feature of the
    contour file."""
    stained_pixels = 0
    total_pixels = 0
    with open(contour_file, 'r') as cf:
        features = json.load(cf)
        for feature in features:
            if feature['geometry']['type'].lower() == 'polygon':
                properties = feature['properties']
                stained_pixels = properties['stained_pixels']
                total_pixels = properties['total_pixels']
    return stained_pixels, total_pixels

def classify_tiles_feature(image_file):
    """Classify the feature type of each tile according to the tissue type model for its contour."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file = os.path.join(image_file_path, contour_file_name)
        tissue_type, _ = get_tissue_type(contour_file)
        print(f'  Classifying {tissue_type} contour {contour_file_name}')
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        command = ['python', 'classify-tiles.py']
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
        command += ['--class', 'feature', '--tiles_dir', tiles_dir]
        model_name = 'model-' + tissue_type
        model_file = os.path.join(MODEL_DIR, model_name)
        command += ['--model', model_file]
        subprocess.run(command)
    return

def classify_tiles_midline(image_file):
    """For obex contours, classify each tile as being on the midline or not."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file = os.path.join(image_file_path, contour_file_name)
        tissue_type, _ = get_tissue_type(contour_file)
        if tissue_type == 'obex':
            contour_file_name_noext = os.path.splitext(contour_file_name)[0]
            print(f'  Classifying midline tiles for contour {contour_file_name}')
            command = ['python', 'classify-tiles.py']
            tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
            command += ['--class', 'midline', '--tiles_dir', tiles_dir]
            model_name = MIDLINE_MODEL
            model_file = os.path.join(MODEL_DIR, model_name)
            command += ['--model', model_file]
            subprocess.run(command)
    return

def get_region_info(contour_file, metadata={}):
    """Returns the region type corresponding to the tissue type of the given contour.
    Also returns the associated probability and area thresholds. If metadata dictionary
    is provided, then the probabilities and thresholds are drawn from there."""
    tissue_type, _ = get_tissue_type(contour_file)
    region_type = None
    prob_threshold = 0.95
    area_threshold = 0
    if tissue_type == 'node':
        region_type = 'follicle'
        prob_threshold = 0.95
        area_threshold = 1310720
        if MINIMUM_PROBABILITY_FOLLICLE_TILE_KEY in metadata:
            prob_threshold = metadata[MINIMUM_PROBABILITY_FOLLICLE_TILE_KEY]
        if MINIMUM_AREA_FOLLICLE_REGION_KEY in metadata:
            area_threshold = metadata[MINIMUM_AREA_FOLLICLE_REGION_KEY]
    if tissue_type == 'obex':
        region_type = 'dorsal_motor_nucleus'
        prob_threshold = 0.95
        area_threshold = 1310720
        if MINIMUM_PROBABILITY_DMN_TILE_KEY in metadata:
            prob_threshold = metadata[MINIMUM_PROBABILITY_DMN_TILE_KEY]
        if MINIMUM_AREA_DMN_REGION_KEY in metadata:
            area_threshold = metadata[MINIMUM_AREA_DMN_REGION_KEY]
    return region_type, prob_threshold, area_threshold

def detect_midline(image_file, enhance=False):
    """Detect midline in obex contours. Write midline to contour JSON and write
    images of different methods."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file = os.path.join(image_file_path, contour_file_name)
        tissue_type, _ = get_tissue_type(contour_file)
        if tissue_type == 'obex':
            contour_file_name_noext = os.path.splitext(contour_file_name)[0]
            print(f'  Detecting midline for contour {contour_file_name}')
            command = ['python', 'detect-midline.py']
            command += ['--image', image_file]
            tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
            command += ['--contour_file', contour_file]
            command += ['--tiles_dir', tiles_dir]
            command += ['--probability', '0.99999']
            if enhance:
                command += ['--enhance']
            subprocess.run(command)
    return

def detect_regions(image_file, metadata):
    """Detect regions in each contour and generate a GeoJSON file for the regions.
    This is done for each detection method."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file = os.path.join(image_file_path, contour_file_name)
        region_type, prob_threshold, area_threshold = get_region_info(contour_file, metadata)
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        print(f'  Detecting {region_type} regions for contour {contour_file_name}')
        command = ['python', 'detect-regions.py']
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
        command += ['--contour_file', contour_file]
        command += ['--class', region_type, '--tiles_dir', tiles_dir]
        command += ['--probability', str(prob_threshold), '--area', str(area_threshold)]
        for detection_method in REGION_DETECTION_METHODS:
            command2 = command + ['--method', detection_method]
            subprocess.run(command2)
    return

def generate_contour_region_images(image_file, metadata, enhance=False):
    """Write an image for each contour with the regions highlighed."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    for contour_file_name in contour_file_names:
        contour_file = os.path.join(image_file_path, contour_file_name)
        region_type, prob_threshold, area_threshold = get_region_info(contour_file, metadata)
        contour_file_base = os.path.splitext(contour_file)[0]
        contour_file_name_noext = os.path.splitext(contour_file_name)[0]
        tiles_dir = os.path.join(image_file_path, contour_file_name_noext + '-tiles')
        command = ['python', 'contour-region-image.py']
        command += ['--image', image_file]
        command += ['--contour', contour_file]
        command += ['--probability', str(prob_threshold)]
        command += ['--tiles_dir', tiles_dir]
        if enhance:
            command += ['--enhance']
        for detection_method in REGION_DETECTION_METHODS:
            regions_file = contour_file_base + '-regions-' + detection_method + '.json'
            print(f'  Generating image for {region_type} regions for contour {contour_file_name} using detection method {detection_method}')
            command2 = command + ['--regions', regions_file]
            subprocess.run(command2)
    return

def write_html_header(html_file, title):
    html_content = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n"
    html_content += "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
    html_content += "    <title>" + title + "</title>\n"
    html_content += "</head>\n<body>\n"
    html_content += "<h1>Image: " + title + "</h1>\n"
    html_file.write(html_content)
    return

def write_html_section(html_file, title, level=2):
    html_content = "<h" + str(level) + ">" + title + "</h" + str(level) + ">\n"
    html_file.write(html_content)
    return

def write_html_footer(html_file):
    html_content = "</body>\n</html>\n"
    html_file.write(html_content)

def write_html_rule(html_file):
    html_content = "<hr>\n"
    html_file.write(html_content)

def write_html_image(html_file, image_file_name, width, br=False):
    html_content = "<img src=\"" + image_file_name + "\" width=" + str(width) + ">"
    if br:
        html_content += " <br>"
    html_content += "\n"
    html_file.write(html_content)
    return

def write_html_region_info(html_file, contour_file, regions_file, br=False):
    # Get region type
    region_type = 'dorsal motor nucleus'
    tissue_type, _ = get_tissue_type(contour_file)
    if tissue_type == 'node':
        region_type = 'follicle'
    # Check for low contrast
    contrast = get_contrast(contour_file)
    if contrast == 'low':
        html_content = "<h3>Low contrast image</h3>"
        if br:
            html_content += " <br>"
        html_content += "\n"
        html_file.write(html_content)
    contour_stained_pixels, contour_total_pixels = get_stained_pixels(contour_file)
    region_staining_info = []
    with open(regions_file, 'r') as rf:
        features = json.load(rf)
        num_regions = len(features)
        num_stained_regions = 0
        for feature in features:
            properties = feature['properties']
            region_stained_pixels = properties['stained_pixels']
            region_total_pixels = properties['total_pixels']
            region_staining_info.append((region_stained_pixels, region_total_pixels))
            if region_stained_pixels > 0: # TODO: higher threshold?
                num_stained_regions += 1
    html_content = "<h3>Found " + str(num_regions) + " " + region_type + " regions (" + str(num_stained_regions) + " exhibit staining)</h3>\n"
    html_content += "<ol>\n"
    for staining_info in region_staining_info:
        percent = 0
        if staining_info[1] > 0:
            percent = 100.0 * staining_info[0] / staining_info[1]
        percent_str = f"{percent:.8f}"
        #html_content += "  <li> " + str(staining_info[0]) + " stained pixels out of " + str(staining_info[1]) + " total pixels in region (" + str(percent) + "%)"
        html_content += "  <li> " + str(staining_info[0]) + " stained pixels in region (" + percent_str + "%)"
    html_content += "</ol>\n"
    percent = 0
    if contour_total_pixels > 0:
        percent = 100.0 * contour_stained_pixels / contour_total_pixels
    percent_str = f"{percent:.8f}"
    #html_content += "<p>" + str(contour_stained_pixels) + " stained pixels out of " + str(contour_total_pixels) + " total pixels in whole tissue (" + str(percent) + "%) </p>\n"
    html_content += "<p>" + str(contour_stained_pixels) + " stained pixels in whole tissue (" + percent_str + "%) </p>\n"
    if br:
        html_content += " <br>"
    html_content += "\n"
    html_file.write(html_content)
    return

def write_html_parameters(html_file, metadata):
    html_content = "<p>Parameters:</p>\n<ul>\n"
    if MINIMUM_PROBABILITY_FOLLICLE_TILE_KEY in metadata:
        html_content += f"<li>Minimum Probability Follicle Tile = {metadata[MINIMUM_PROBABILITY_FOLLICLE_TILE_KEY]}</li>\n"
    if MINIMUM_PROBABILITY_DMN_TILE_KEY in metadata:
        html_content += f"<li>Minimum Probability DMN Tile = {metadata[MINIMUM_PROBABILITY_DMN_TILE_KEY]}</li>\n"
    if MINIMUM_AREA_FOLLICLE_REGION_KEY in metadata:
        html_content += f"<li>Minimum Area Follicle Region = {metadata[MINIMUM_AREA_FOLLICLE_REGION_KEY]}</li>\n"
    if MINIMUM_AREA_DMN_REGION_KEY in metadata:
        html_content += f"<li>Minimum Area DMN Region = {metadata[MINIMUM_AREA_DMN_REGION_KEY]}</li>\n"
    if MINIMUM_PERCENT_STAINED_PIXELS_NODE_KEY in metadata:
        html_content += f"<li>Minimum Percent Stained Pixels Node = {metadata[MINIMUM_PERCENT_STAINED_PIXELS_NODE_KEY]}%</li>\n"
    if MINIMUM_PERCENT_STAINED_PIXELS_OBEX_KEY in metadata:
        html_content += f"<li>Minimum Percent Stained Pixels Obex = {metadata[MINIMUM_PERCENT_STAINED_PIXELS_OBEX_KEY]}%</li>\n"
    if MINIMUM_NUMBER_FOLLICLES_KEY in metadata:
        html_content += f"<li>Minimum Number of Follicles = {metadata[MINIMUM_NUMBER_FOLLICLES_KEY]}</li>\n"
    #if SHOW_DETAILS_FLAG_KEY in metadata:
    #    html_content += f"<li>Show Details Flag = {metadata[SHOW_DETAILS_FLAG_KEY]}</li>\n"    
    html_content += "</ul>\n"
    html_file.write(html_content)
    return

def write_html_basic_results(image_file, index_html_file, metadata):
    """Output HTML for basic analysis results showing each contour's designation."""
    image_file_path = os.path.dirname(image_file)
    contour_file_names = filter_contour_files(image_file)
    write_html_section(index_html_file, "Basic Results")
    html_content = "<p>No tissue detected.</p>\n"
    if len(contour_file_names) > 0:
        html_content = "<ol>\n"
        for contour_file_name in contour_file_names:
            contour_file = os.path.join(image_file_path, contour_file_name)
            contour_file_noext = os.path.splitext(contour_file)[0]
            regions_file = contour_file_noext + '-regions-' + DEFAULT_DETECTION_METHOD + '.json'
            result = "Unknown"
            stained_pixels, total_pixels = get_stained_pixels(contour_file)
            percent_stained_pixels = 0.0
            if total_pixels > 0:
                percent_stained_pixels = 100.0 * (stained_pixels / total_pixels)
            tissue_type, _ = get_tissue_type(contour_file)
            num_regions = 0
            with open(regions_file, 'r') as rf:
                features = json.load(rf)
                num_regions = len(features)
            if tissue_type == 'node':
                if percent_stained_pixels > metadata[MINIMUM_PERCENT_STAINED_PIXELS_NODE_KEY]:
                    result = "Detected"
                elif num_regions < metadata[MINIMUM_NUMBER_FOLLICLES_KEY]:
                    result = "Insufficient follicles"
                else:
                    result = "Not detected"
            if tissue_type == 'obex':
                if percent_stained_pixels > metadata[MINIMUM_PERCENT_STAINED_PIXELS_OBEX_KEY]:
                    result = "Detected"
                elif num_regions > 0:
                    result = "Not detected"
                else:
                    result = "Location"
            html_content += f"  <li>{tissue_type.capitalize()}: {result}</li>\n"
        html_content += "</ol>\n"
    index_html_file.write(html_content)
    return

def write_html_detailed_results(image_file, html_path, index_html_file, metadata):
    """Output HTML for detailed analysis results."""
    image_file_path = os.path.dirname(image_file)
    image_file_base = os.path.splitext(image_file)[0]
    contour_file_names = filter_contour_files(image_file)
    index_html_file.write("<details>\n<summary>Show Details</summary>\n")
    if len(contour_file_names) > 0:
        all_contours_file = image_file_base + '-contours.png'
        shutil.copy(all_contours_file, html_path)
        write_html_section(index_html_file, "Detected Tissue Contours")
        write_html_image(index_html_file, os.path.basename(all_contours_file), 600, br=True)
        contour_file_names = filter_contour_files(image_file)
        contour_num = 1
        for contour_file_name in contour_file_names:
            write_html_rule(index_html_file)
            contour_file = os.path.join(image_file_path, contour_file_name)
            contour_file_noext = os.path.splitext(contour_file)[0]
            # Copy image files to HTML directory
            shutil.copy(contour_file_noext + '.png', html_path)
            shutil.copy(contour_file_noext + '-distribution.png', html_path)
            for detection_method in REGION_DETECTION_METHODS:
                file = contour_file_noext + '-regions-' + detection_method + '.png'
                shutil.copy(file, html_path)
                file = contour_file_noext + '-regions-' + detection_method + '-prob.png'
                shutil.copy(file, html_path)
                file = contour_file_noext + '-regions-' + detection_method + '-heatmap.png'
                shutil.copy(file, html_path)
            contour_file_name_noext = os.path.splitext(contour_file_name)[0]
            tissue_type, certainty = get_tissue_type(contour_file)
            write_html_section(index_html_file, f"Contour {contour_num}: {tissue_type} (certainty = {certainty:.6f})")
            write_html_image(index_html_file, contour_file_name_noext + '-distribution.png', 400, br=True)
            if tissue_type == 'obex':
                for suffix in ['-midline-concavity', '-midline-tiles']:
                    file = contour_file_noext + suffix + '.png'
                    shutil.copy(file, html_path)
                write_html_section(index_html_file, "Midline Detection (concavity method, DL tile classification method)", level=3)
                write_html_image(index_html_file, contour_file_name_noext + '-midline-concavity.png', 400, br=False)
                write_html_image(index_html_file, contour_file_name_noext + '-midline-tiles.png', 400, br=True)
            region_type, prob_threshold, area_threshold = get_region_info(contour_file, metadata)
            for detection_method in REGION_DETECTION_METHODS:
                if ('diagonal' in detection_method) and ('_' in detection_method):
                    factor = float(detection_method.split('_')[1])
                    area_threshold = int(area_threshold * factor)
                write_html_section(index_html_file, f"Region Detection [Method: {detection_method.upper()}, Prob > {prob_threshold:.3f}, Area > {area_threshold}] (regions, tiles, heatmap)", level=3)
                region_file_name_suffix = contour_file_name_noext + '-regions-' + detection_method
                write_html_image(index_html_file, region_file_name_suffix + '.png', 400, br=False)
                write_html_image(index_html_file, region_file_name_suffix + '-prob.png', 400, br=False)
                write_html_image(index_html_file, region_file_name_suffix + '-heatmap.png', 400, br=False)
                region_file_suffix = contour_file_noext + '-regions-' + detection_method
                write_html_region_info(index_html_file, contour_file, region_file_suffix + '.json', br=True)
            contour_num += 1
    else:
        index_html_file.write("<p>No tissue detected.</p>\n")
    index_html_file.write("</details>\n")
    return

def write_html(image_file, metadata):
    """Write all analysis info in HTML format."""
    image_file_path = os.path.dirname(image_file)
    image_file_base = os.path.splitext(image_file)[0]
    # Create clean HTML directory
    html_path = image_file_base + '-html'
    if os.path.exists(html_path):
        shutil.rmtree(html_path)
    os.mkdir(html_path)
    index_html_file_name = os.path.join(html_path, 'index.html')
    with open(index_html_file_name, 'w') as index_html_file:
        write_html_header(index_html_file, os.path.basename(image_file_base))
        write_html_basic_results(image_file, index_html_file, metadata)
        show_details_flag = True
        if metadata:
            write_html_parameters(index_html_file, metadata)
            if SHOW_DETAILS_FLAG_KEY in metadata:
                show_details_flag = metadata[SHOW_DETAILS_FLAG_KEY]
        #if show_details_flag:
        write_html_detailed_results(image_file, html_path, index_html_file, metadata)
        write_html_footer(index_html_file)
    return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', dest='image_file', type=str, required=True)
    parser.add_argument('--enhance', action='store_true')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--metadata', dest='metadata_file', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    image_file = args.image_file
    enhance = args.enhance
    background = args.background
    metadata_file = args.metadata_file
    metadata = {}
    if metadata_file:
        with open(metadata_file, 'r') as mf:
            metadata = json.load(mf)
    segment_image(image_file, enhance)
    generate_tiles(image_file, enhance, background)
    classify_tiles_tissue(image_file)
    classify_contours(image_file)
    classify_tiles_feature(image_file)
    classify_tiles_midline(image_file)
    detect_midline(image_file, enhance)
    detect_regions(image_file, metadata)
    generate_contour_region_images(image_file, metadata, enhance)
    write_html(image_file, metadata)
    return

if __name__ == "__main__":
    main()
