# segment.py --image <svs_image_file> --enhance
#
# Segment a scaled version of the input SVS slide image into tissue and non-tissue
# components. Identify contours in image that outline major tissue components. For
# each tissue component generate an image with that component against a white
# background and generate a GeoJSON file with the coordinates of the contour. Also
# generate a copy of the original image with all tissue contours highlighted in
# different colors.
#
# Segments are checked for low contrast, and this information is written to the
# GeoJSON file. If the --enhance argument is given, then contour images are enhanced
# for improved contrast using skimage.exposure.equalize_adapthist method.
#
# Segmentation is based on the marching squares algorithm, which is implemented
# in the scikit-image find_contours method.
#
# Author: Lawrence Holder, Washington State University

import numpy as np
from openslide import OpenSlide
from PIL import Image, ImageDraw
import skimage
import sys
import os
import math
import ntpath
import geojson
import warnings
import argparse

SCALE_FACTOR = 32 # For working with scaled down versions of slides
BORDER_FILTER = 0.01 # fraction near border for filtering contour points
MAX_RATIO_CONTOUR_AREAS = 5 # if area(contour_larger)/area(contour_smaller) > this, then filter

def segment_contours(image):
    """Find and process contours."""
    # Crop image and convert to grayscale and crop
    cropped_image = crop_image(image)
    gray_image = skimage.color.rgb2gray(cropped_image)
    threshold_value = 0.95  # Adjust this value as needed to filter slide background
    binary_image = gray_image < threshold_value
    #b_image = binary_image * 255
    #skimage.io.imsave('image.png', b_image.astype(np.uint8))
    # Find contours of tissue regions and process
    contours = skimage.measure.find_contours(binary_image) #, 0.95)
    # Filter contours to ensure closed, non-degenerate contours (convert to list)
    contours = filter_contours(contours)
    #write_all_contours_image('temp', cropped_image, contours)
    # Get contour areas and sort them by area
    contour_areas = [compute_contour_area(c) for c in contours]
    contours_areas_sorted = sorted(zip(contours, contour_areas), key=lambda x: x[1], reverse=True)
    # Get largest contours (if area drops by more the 50%, then stop)
    largest_contours_and_areas = []
    #contour_area_prev = 0
    contour_area_prev = contours_areas_sorted[0][1]
    for contour, contour_area in contours_areas_sorted:
        #print('contour length = ' + str(len(contour)))
        #print('contour area = ' + str(contour_area))
        if ((contour_area_prev / contour_area) > MAX_RATIO_CONTOUR_AREAS):
            break
        largest_contours_and_areas.append((contour, contour_area))
        #contour_area_prev = contour_area
    # Smooth contours (filtering intermittent points to simplify contour)
    sm_contours_and_areas = [(smooth_contour(c),a) for (c,a) in largest_contours_and_areas]
    sp_contours = split_contours(sm_contours_and_areas, binary_image)
    final_contours = uncrop_contours(sp_contours, image)
    return final_contours

def uncrop_contours(contours, image):
    """Shifts coordinates of contours according to the original cropping of the image,
    so coordinates align with original image. This should be based on the crop_image
    method (in reverse)."""
    h,w,_ = image.shape
    border = math.floor(h * BORDER_FILTER)
    uncropped_contours = []
    for contour in contours:
        # Add border to all coordinates
        contour1 = np.array(contour) + border
        uncropped_contour = contour1.tolist()
        uncropped_contours.append(uncropped_contour)
    return uncropped_contours

def compute_contour_area(contour):
    contour = np.array(contour)
    x = contour[:,1]
    y = contour[:,0]
    contour_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return contour_area

def split_contours(contours_and_areas, image):
    """If first contour takes up more than 50% of the image,
       then split the largest contour at potential overlap point."""
    height, width = image.shape
    half_area = width * height / 2
    first_contour = contours_and_areas[0][0]
    first_contour_area = contours_and_areas[0][1]
    while (first_contour_area > half_area) and (len(first_contour) > 3):
        #print('contour area ' + str(first_contour_area) + ' > half area ' + str(half_area))
        contour_and_area_1, contour_and_area_2 = split_contour(first_contour)
        area1 = contour_and_area_1[1]
        area2 = contour_and_area_2[1]
        if (area1 == 0.0) or (area2 == 0.0):
            break
        if area1 > area2:
            if (area1 / area2) > MAX_RATIO_CONTOUR_AREAS:
                break # newly split off contour is too small
            else:
                contours_and_areas = [contour_and_area_1, contour_and_area_2] + contours_and_areas[1:]
        else:
            if (area2 / area1) > MAX_RATIO_CONTOUR_AREAS:
                break # newly split off contour is too small
            else:
                contours_and_areas = [contour_and_area_2, contour_and_area_1] + contours_and_areas[1:]
        first_contour = contours_and_areas[0][0]
        first_contour_area = contours_and_areas[0][1]
    contours = [ca[0] for ca in contours_and_areas]
    return contours

def split_contour(contour):
    """Split contour into two contours based on pair of points maximizing the difference
       in their order in the contour and minimizing their distance. Return the two
       (contour, contour_area) pairs."""
    len_contour = len(contour)
    #print('split contour (len = ' + str(len_contour) + ')')
    max_score = 0.0
    max_index1 = max_index2 = 0
    for index1 in range(len_contour):
        for index2 in range(index1+1, len_contour):
            point1 = contour[index1]
            point2 = contour[index2]
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            offset = min(index2 - index1, index1 + (len_contour - index2))
            score = offset - distance
            #if distance > 0:
            #    score = offset / distance
            #print(str([index1,index2,offset,distance,score]))
            if score > max_score:
                max_score = score
                max_index1 = index1
                max_index2 = index2
    #print('split contour (len=' + str(len_contour) + ') at ' + str([max_index1,max_index2]))
    area1 = area2 = 0.0
    contour1 = contour[max_index1:max_index2+1]
    if len(contour1) > 1:
        contour1.append(contour[max_index1]) # close contour
        area1 = compute_contour_area(contour1)
    contour2 = contour[max_index2+1:] + contour[:max_index1+1]
    if len(contour2) > 1:
        contour2.append(contour[max_index2+1]) # close contour
        area2 = compute_contour_area(contour2)
    ca1 = (contour1, area1)
    ca2 = (contour2, area2)
    return ca1, ca2

def filter_contours(contours):
    """Ensure contours are non-degenerate and closed. Input contours are numpy arrays.
    Returned list of contours are no longer numpy arrays."""
    filtered_contours = []
    for contour in contours:
        contour = contour.tolist()
        if len(contour) > 2:
            # Ensure a closed contour
            if not (contour[0] == contour[-1]):
                contour.append(contour[0])
            filtered_contours.append(contour)
    return filtered_contours

def smooth_contour(contour, scale_factor=50):
    """Smooth contour by keeping only every scale_factor vertex."""
    # Remove last vertex
    sm_contour = contour[:-1]
    sm_contour = sm_contour[::scale_factor]
    # Put closing vertex back in
    sm_contour.append(sm_contour[0])
    return sm_contour

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

def get_scaled_image_pil(slide):
    w, h = slide.dimensions
    w = math.floor(w / SCALE_FACTOR)
    h = math.floor(h / SCALE_FACTOR)
    im = slide.get_thumbnail((w,h))
    im = im.convert('RGB') # remove alpha channel, if present
    return im

def get_scaled_image_np(slide):
    # Note numpy image shape = (height,width)
    im = get_scaled_image_pil(slide)
    return np.array(im)

def print_properties(slide):
    #for key,value in slide.properties.items():
    #    print(str(key) + ' = ' + str(value))
    num_levels = slide.level_count
    levels_dimensions = [slide.level_dimensions[i] for i in range(num_levels)]
    downsampling_factors = [slide.level_downsamples[i] for i in range(num_levels)]
    print("Number of levels:", num_levels)
    print("Dimensions of each level:", levels_dimensions)
    print("Downsampling factors of each level:", downsampling_factors)
    return

def write_all_contours_image(filebase, image, contours):
    """Draw the contours on the given image and save to file. Uses Pillow."""
    contour_image = Image.fromarray(image)
    contour_draw = ImageDraw.Draw(contour_image)
    colors = ['red', 'green', 'blue', 'yellow']
    line_thickness = 2
    color_index = 0
    for contour in contours:
        y1,x1 = contour[0] # in yx format, but pillow uses xy
        for point in contour[1:]:
            y2,x2 = point
            contour_draw.line([(x1,y1),(x2,y2)], fill=colors[color_index], width=line_thickness)
            x1,y1 = x2,y2
        color_index = (color_index + 1) % len(colors)
    img = np.array(contour_image)
    save_image_no_warning(filebase + '-contours.png', img)
    return

def write_contour_as_geojson(filebase, contour_yx, low_contrast):
    """Writes contour, given as (y,x) points, to GeoJSON file in (x,y) format for input to QuPath."""
    contour_xy = [[x,y] for y,x in contour_yx]
    properties = {
        "objectType": "annotation",
        "classification": {
            "name": "tissue",
            "color": [255, 0, 0]
        },
        "contrast": ('low' if low_contrast else 'normal')
    }
    feature = geojson.Feature(geometry=geojson.Polygon([contour_xy]), properties=properties)
    feature_arr = [feature]
    with open(filebase + '.json', 'w') as geojson_file:
        geojson.dump(feature_arr, geojson_file, indent=2)
    return

def write_contours_and_images(filebase, image, contours, enhance=False):
    """For each contour, write scaled-down image with contour on white background. If
    enhance is true, then improve contrast of image. Also, write contour points to JSON
    file, but scaled back up to original image size."""
    contour_num = 0
    for contour in contours:
        contour_num += 1
        # Generate contour image
        contour_image = extract_contour_image(image, contour)
        low_contrast = skimage.exposure.is_low_contrast(contour_image, fraction_threshold=0.05)
        filebase1 = filebase + '-contour-' + str(contour_num)
        if enhance:
            contour_image = skimage.exposure.equalize_adapthist(contour_image) # removes alpha channel, returns floats
            contour_image = contour_image * 255
            contour_image = contour_image.astype(np.uint8)
        #save_image_no_warning(filebase1 + '.png', contour_image)
        skimage.io.imsave(filebase1 + '.png', contour_image)
        # Write contour points
        contour = np.array(contour)
        contour = contour * SCALE_FACTOR
        contour = contour.astype(int)
        contour_yx = contour.tolist()
        write_contour_as_geojson(filebase1, contour_yx, low_contrast)
    return

def save_image_no_warning(file, image):
    """Save image and suppress warnings.
    Most of these images generate a low-contrast warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(file, image)
    return

def crop_image(image):
    """Mainly to remove banding at the borders of WSI images."""
    h,w,_ = image.shape
    border = math.floor(h * BORDER_FILTER)
    cropped_image = image[border:h-border,border:w-border]
    return cropped_image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_file', type=str, required=True)
    parser.add_argument('--enhance', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    # Read svs image
    image_file = args.image_file
    print('Processing image: ' + image_file)
    filebase, _ = os.path.splitext(image_file)
    slide = OpenSlide(image_file)
    w, h = slide.dimensions
    print('  original dimensions: ' + str(w) + 'w x ' + str(h) + 'h')
    #print_properties(slide)
    scaled_image = get_scaled_image_np(slide)
    h,w,_ = scaled_image.shape
    print('  scaled dimensions: ' + str(w) + 'w x ' + str(h) + 'h')
    contours = segment_contours(scaled_image)
    print('  found ' + str(len(contours)) + ' contours')
    write_all_contours_image(filebase, scaled_image, contours)
    write_contours_and_images(filebase, scaled_image, contours, enhance=args.enhance)
    print('  wrote ' + str(len(contours)) + ' contours and images')
    return

if __name__ == "__main__":
    main()
