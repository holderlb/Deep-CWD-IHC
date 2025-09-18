# detect-regions.py --contour_file <contour_file> --tiles_dir <tiles_dir>
#   --class <class_name> [--area 0.0] [--probability 0.5] [--method 'diagonal']
#
# Reads in the <class_name> tiles for the given contour from the 'tiles.csv'
# file in <tiles_dir> and identifies contiguous regions of these tiles. The regions
# are written as polygons to a QuPath-compatible GeoJSON-formatted annotations file
# '<contour>.region_annotations.json'. The --class must be one of the elements in
# the CLASS_NAMES list below. If the --probability argument is provided, then
# the prediction probability must be at least this high for some methods (see below).
# The default probability threshold is 0.5. If the --area argument is provided, then
# a region's area must be at least this size. The area is given in pixels in the
# original slide resolution. The default area threshold is 0.0.
#
# There are several methods available to compute regions (see below). The --method
# argument can be used to select the desired method. Note that the adjacent, diagonal,
# and cellpose approaches use only tiles whose probability is at least that given by
# the --probability argument.
#   adjacent1: combines regions R1 and R2 if a single tile in R1 is
#     adjacent to a single tile in R2.
#   adjacent2: combines two regions R1 and R2 where two adjacent tiles in R1
#     are adjacent to two adjacent tiles in R2.
#   adjacent4: adds tile to region if 4 adjacent tiles also in region.
#   diagonal: combines two regions R1 and R2 if a tile in R1 is adjacent to
#     a tile in R2, and R2 has a tile diagonal from the tile in R1.
#   cellpose:
#   watershed:
#   graph:
#   gaussian:
#
# Author: Lawrence Holder, Washington State University

import os
import skimage
import sklearn
import numpy as np
import argparse
import json
import csv
import shapely
from shapely.geometry import LineString, Polygon, mapping
import geojson
from PIL import ImageColor
import scipy
from cellpose import models

TILES_CSV_FILE = 'tiles.csv'
CLASS_NAMES = ['dorsal_motor_nucleus', 'follicle']
REGION_DETECTION_METHODS = ['adjacent1', 'adjacent2', 'adjacent4', 'diagonal', 'diagonal_1.0', 'diagonal_0.50', 'diagonal_0.25',
                            'cellpose', 'watershed', 'graph', 'graph_0.5', 'graph_0.6', 'graph_0.7', 'graph_0.8', 'graph_0.89',
                            'gaussian', 'gaussian_10', 'gaussian_20', 'gaussian_30', 'gaussian_40', 'gaussian_50']
REGION_DETECTION_METHODS_WITH_THRESHOLD = ['adjacent1', 'adjacent2', 'adjacent4', 'diagonal',
                                           'diagonal_1.0', 'diagonal_0.50', 'diagonal_0.25'] # min area threshold reduced by given argument

class CustomPolygon:
    def __init__(self, polygon, stained_pixels=0, total_pixels=0):
        self.polygon = polygon
        self.stained_pixels = stained_pixels
        self.total_pixels = total_pixels

    def __repr__(self):
        return (f"CustomPolygon(stained_pixels={self.stained_pixels}, total_pixels={self.total_pixels}, "
                f"polygon={self.polygon.wkt})")
    
    def area(self):
        return self.polygon.area
    
    def distance(self, line):
        return self.polygon.distance(line)
    
    def coordinates(self):
        return mapping(self.polygon)['coordinates']
    

def read_tiles(tiles_dir, class_name):
    tiles_file_name = os.path.join(tiles_dir, TILES_CSV_FILE)
    tiles = []
    with open(tiles_file_name, 'r') as tiles_file:
        for row in csv.DictReader(tiles_file):
            predicted_class = row['feature_prediction']
            if (predicted_class == class_name):
                x = int(row['x'])
                y = int(row['y'])
                w = int(row['width'])
                h = int(row['height'])
                sp = int(row['stained_pixels'])
                probability = float(row['feature_probability'])
                tile = [x, y, w, h, sp, probability]
                tiles.append(tile)
    return tiles

def read_contour_midline(contour_file_name):
    with open(contour_file_name, 'r') as contour_file:
        features = json.load(contour_file)
        points = []
        for feature in features:
            properties = feature['properties']
            classification = properties['classification']
            class_name = classification['name']
            if class_name == 'midline':   
                geometry = feature['geometry']
                coordinates = geometry['coordinates']
                for point in coordinates:
                    x = point[0]
                    y = point[1]
                    points.append((x,y))
    line = None
    if points:
        line = LineString(points)
    return line

def compute_regions(tiles, args):
    method = args.detection_method
    if '_' in method: # argument included after '_'
        method_base = method.split('_')[0]
        arg = float(method.split('_')[1])
        method_function_name = 'compute_regions_' + method_base
        region_ids = globals()[method_function_name](tiles, arg)
    else:
        method_function_name = 'compute_regions_' + method
        region_ids = globals()[method_function_name](tiles)
    return region_ids

def compute_regions_adjacent1(tiles):
    """Find sets of contiguous tiles."""
    # Build initial single-tile regions
    n = len(tiles)
    region_ids = list(range(n))
    # Merge regions with pair of adjacent tiles
    for i in range(n):
        for j in range(i+1,n):
            ri = region_ids[i]
            rj = region_ids[j]
            if (ri != rj) and (adjacent(tiles[i], tiles[j])):
                x1,x2 = ri,rj
                if ri > rj:
                    x1,x2 = rj,ri
                for k in range(len(region_ids)):
                    if region_ids[k] == x2:
                        region_ids[k] = x1
    return region_ids

def compute_regions_adjacent2(tiles):
    """Find sets of contiguous tiles, but no single-tile bridges. Two regions can
    merge only if there are two adjacent tiles from one region adjacent to two
    adjacent tiles from another region."""
    # Build initial single-tile regions
    n = len(tiles)
    region_ids = list(range(n))
    region_sizes = [1] * n
    # Consider merging regions with pair of adjacent tiles
    merge_happened = True
    while merge_happened:
        merge_happened = False
        for i in range(n):
            for j in range(i+1,n):
                ri = region_ids[i]
                rj = region_ids[j]
                if (ri != rj) and (adjacent(tiles[i], tiles[j])):
                    ri_size = region_sizes[ri]
                    rj_size = region_sizes[rj]
                    if (ri_size > 1) and (rj_size > 1):
                        if not another_adjacency(tiles, region_ids, i, j):
                            continue
                    # Regions can be merged, so merge them
                    x1,x2 = ri,rj
                    if ri > rj:
                        x1,x2 = rj,ri
                    for k in range(len(region_ids)):
                        if region_ids[k] == x2:
                            region_ids[k] = x1
                            region_sizes[x1] += 1
                    region_sizes[x2] = 0
                    merge_happened = True
    return region_ids

def compute_regions_diagonal(tiles, _area_threshold_factor=0.0):
    """Find sets of contiguous tiles, but no single-tile bridges. When merging
    two regions based on two adjacent tiles, if a region has more than one tile,
    then it must also have a tile diagonal from the adjacent tile in the other region.
    The _area_threshold_factor argument is ignored; this factor is applied in
    compute_region_polygons."""
    # Build initial single-tile regions
    n = len(tiles)
    region_ids = list(range(n))
    region_sizes = [1] * n
    # Consider merging regions with pair of adjacent tiles
    merge_happened = True
    while merge_happened:
        merge_happened = False
        for i in range(n):
            for j in range(i+1,n):
                ri = region_ids[i]
                rj = region_ids[j]
                if (ri != rj) and (adjacent(tiles[i], tiles[j])):
                    ri_size = region_sizes[ri]
                    rj_size = region_sizes[rj]
                    if ri_size > 1 and (not diagonal_exists(tiles, region_ids, ri, i, j)):
                        continue
                    if rj_size > 1 and (not diagonal_exists(tiles, region_ids, rj, j, i)):
                        continue
                    # Regions can be merged, so merge them
                    x1,x2 = ri,rj
                    if ri > rj:
                        x1,x2 = rj,ri
                    for k in range(len(region_ids)):
                        if region_ids[k] == x2:
                            region_ids[k] = x1
                            region_sizes[x1] += 1
                    region_sizes[x2] = 0
                    merge_happened = True
    return region_ids

def diagonal_exists(tiles, region_ids, region_id, region_tile_index, other_tile_index):
    """Checks that there exists a tile in the region_id region that is adjacent
    to the region_tile and diagonal from the other_tile. The region_tile and
    other_tile are assumed to be adjacent. Also assumes tiles are same size
    and square, i.e., width == height."""
    region_x, region_y, width = tiles[region_tile_index][:3]
    other_x, other_y = tiles[other_tile_index][:2]
    if (region_x == other_x): # other tile above/below region tile
        diag1_x, diag1_y = (region_x - width, region_y)
        diag2_x, diag2_y = (region_x + width, region_y)
    if (region_y == other_y): # other tile left/right region tile
        diag1_x, diag1_y = (region_x, region_y - width)
        diag2_x, diag2_y = (region_x, region_y + width)
    for i in range(len(tiles)):
        if region_ids[i] == region_id:
            x, y = tiles[i][:2]
            if (x,y) == (diag1_x, diag1_y):
                return True
            if (x,y) == (diag2_x, diag2_y):
                return True
    return False

def compute_regions_adjacent4(tiles):
    """Uses the scipy.ndimage.label method to identify regions in the tiles.
    Tiles are in format [x,y,width,height,stain,probability]."""
    # Normalize tile positions to a grid
    tile_dict = {(t[0] // t[2], t[1] // t[3]): idx for idx, t in enumerate(tiles)}
    max_x = max(t[0] // t[2] for t in tiles) + 1
    max_y = max(t[1] // t[3] for t in tiles) + 1
    grid = np.zeros((max_x, max_y), dtype=int)
    for t in tiles:
        grid[t[0] // t[2], t[1] // t[3]] = 1
    # SciPy label method using default structure of 4 adjacent pixels (tiles)
    # Returns grid with feature labels assigned to each position
    labeled_array, num_features = scipy.ndimage.label(grid) # Default structure is 4 adjacent tiles
    # Assign region IDs to each tile
    region_ids = [-1] * len(tiles)
    for (norm_x, norm_y), tile_idx in tile_dict.items():
        region_ids[tile_idx] = labeled_array[norm_x, norm_y]
    return region_ids

def compute_regions_cellpose_binary(tiles):
    """Uses the CellPose method on binary mask to identify regions in the tiles.
    If you use this version, be sure to include cellpose in the list of methods
    where the probability threshold is first applied on the tiles.
    Tiles are in format [x,y,width,height,stain,probability]."""
    tile_dict = {(t[0] // t[2], t[1] // t[3]): idx for idx, t in enumerate(tiles)}
    max_x = max(t[0] // t[2] for t in tiles) + 1
    max_y = max(t[1] // t[3] for t in tiles) + 1
    binary_mask = np.zeros((max_x, max_y), dtype=np.uint8)
    for t in tiles:
        binary_mask[t[0] // t[2], t[1] // t[3]] = 1
    model = models.Cellpose(model_type='nuclei')
    masks, _, _, _ = model.eval(binary_mask, diameter=None, channels=[0, 0])
    # Assign region IDs to each tile
    region_ids = [-1] * len(tiles)
    for (norm_x, norm_y), tile_idx in tile_dict.items():
        region_ids[tile_idx] = masks[norm_x, norm_y]
    return region_ids

def compute_regions_cellpose(tiles):
    """Uses the CellPose method on grayscale mask to identify regions in the tiles.
    Tiles are in format [x,y,width,height,stain,probability]."""
    tile_dict = {(t[0] // t[2], t[1] // t[3]): idx for idx, t in enumerate(tiles)}
    max_x = max(t[0] // t[2] for t in tiles) + 1
    max_y = max(t[1] // t[3] for t in tiles) + 1
    mask = np.zeros((max_x, max_y), dtype=np.uint8)
    for t in tiles:
        mask[t[0] // t[2], t[1] // t[3]] = int(t[5] * 255)
    model = models.Cellpose(model_type='nuclei')
    masks, _, _, _ = model.eval(mask, diameter=None, channels=[0, 0])
    # Assign region IDs to each tile
    region_ids = [-1] * len(tiles)
    for (norm_x, norm_y), tile_idx in tile_dict.items():
        region_ids[tile_idx] = masks[norm_x, norm_y]
    return region_ids

def compute_regions_watershed(tiles):
    """Uses the skimage.segmentation.watershed method to identify regions in the tiles.
    Tiles are in format [x,y,width,height,stain,probability]."""
    tile_dict = {(t[0] // t[2], t[1] // t[3]): idx for idx, t in enumerate(tiles)}
    max_x = max(t[0] // t[2] for t in tiles) + 1
    max_y = max(t[1] // t[3] for t in tiles) + 1
    probability_grid = np.zeros((max_x, max_y))
    for t in tiles:
        probability_grid[t[0] // t[2], t[1] // t[3]] = t[5]
    local_max_coords = skimage.feature.peak_local_max(probability_grid, footprint=np.ones((3, 3)))
    mask = np.zeros(probability_grid.shape, dtype=bool)
    mask[tuple(local_max_coords.T)] = True
    markers, _ = scipy.ndimage.label(mask)
    segmented_regions = skimage.segmentation.watershed(-probability_grid, markers)
    # Assign region IDs to each tile
    region_ids = [-1] * len(tiles)
    for (norm_x, norm_y), tile_idx in tile_dict.items():
        region_ids[tile_idx] = segmented_regions[norm_x, norm_y]
    return region_ids

def compute_regions_graph(tiles, damping=0.5):
    """Uses the sklearn.cluster.AffinityPropagation method to identify regions in the tiles.
    Tiles are in format [x,y,width,height,stain,probability]."""
    # Normalize positions
    coords = np.array([(t[0] // t[2], t[1] // t[3], t[5]) for t in tiles])
    clustering = sklearn.cluster.AffinityPropagation(damping=damping, max_iter=500, random_state=42).fit(coords)
    region_ids = clustering.labels_.tolist()
    return region_ids

def compute_regions_gaussian(tiles, n_components=50):
    """Uses the sklearn.mixture.GaussianMixture method to identify regions in the tiles.
    Requires a guess as to the maximum number of clusters, so not as useful.
    Tiles are in format [x,y,width,height,stain,probability]."""
    n_components = int(n_components)
    coords = np.array([(t[0] // t[2], t[1] // t[3], t[5]) for t in tiles])
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(coords)
    region_ids = gmm.predict(coords).tolist()
    return region_ids

def compute_polygon_from_tiles(tiles):
    polygons = []
    stained_pixels = 0
    total_pixels = 0
    for tile in tiles:
        x,y,w,h,sp = tile[:5]
        x1, y1 = x, y
        x2, y2 = x + w, y
        x3, y3 = x + w, y + h
        x4, y4 = x, y + h
        polygon = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        polygons.append(polygon)
        stained_pixels += sp
        total_pixels += (w * h)
    #final_polygon = polygons[0] if len(polygons) == 1 else polygons[0].union(polygons[1:])
    final_polygon = shapely.unary_union(polygons)
    # Make sure final polygon is one polygon hull
    #final_polygon = union_polygon.convex_hull
    if isinstance(final_polygon, shapely.geometry.MultiPolygon):
        final_polygon = shapely.concave_hull(final_polygon, ratio=0.5)
    final_custom_polygon = CustomPolygon(final_polygon, stained_pixels, total_pixels)
    return final_custom_polygon

def adjacent(tile1, tile2):
    """Return true if tiles are adjacent, i.e., share a side."""
    x1,y1,w1,h1 = tile1[:4]
    x2,y2,w2,h2 = tile2[:4]
    # tile1 left of tile2
    if ((x1+w1) == x2) and (y1 == y2):
        return True
    # tile1 right of tile2
    if ((x2+w2) == x1) and (y1 == y2):
        return True
    # tile1 above tile2
    if ((y1+h1) == y2) and (x1 == x2):
        return True
    # tile1 below tile2
    if ((y2+w2) == y1) and (x1 == x2):
        return True
    return False

def another_adjacency(tiles, region_ids, region1_tile_index, region2_tile_index):
    """Checks if the two regions have another pair of adjacent tiles beyond the given pair
    and these tiles must be adjacent to the given pair tile in their region."""
    n = len(tiles)
    region1 = region_ids[region1_tile_index]
    region2 = region_ids[region2_tile_index]
    tile1 = tiles[region1_tile_index]
    tile2 = tiles[region2_tile_index]
    for i in range(n):
        for j in range(n):
            if (region_ids[i] == region1) and (region_ids[j] == region2):
                if (i != region1_tile_index) and (j != region2_tile_index):
                    if (adjacent(tiles[i], tiles[j]) and adjacent(tiles[i], tile1) and adjacent(tiles[j], tile2)):
                        return True
    return False

def filter_dmn_regions(contour_file_name, region_polygons):
    """If contour file contains the midline, then filter region polygons
    that are too far away from the line."""
    midline = read_contour_midline(contour_file_name)
    close_polygons = []
    if midline:
        print('Filtering DMN regions...')
        print('  midline = ' + str(midline))
        distance_threshold = compute_distance_threshold(midline)
        print('  distance threshold = ' + str(distance_threshold))
        print('  checking ' + str(len(region_polygons)) + ' polygons')
        for polygon in region_polygons:
            distance = polygon.distance(midline)
            print('  distance = ' + str(distance))
            if distance < distance_threshold:
                close_polygons.append(polygon)
    return close_polygons

def compute_distance_threshold(line):
    """For now, the distance threshold is half the length of the line, which by
    construction (see classify_contour.py) is about half the width of the contour."""
    return line.length # / 2.0

def get_color(class_name):
    color_name = 'red' # default
    if class_name == 'dorsal_motor_nucleus':
        color_name = 'green'
    if class_name == 'follicle':
        color_name = 'blue'
    color_rgb = ImageColor.getcolor(color_name, "RGB")
    return color_rgb

def write_regions_as_geojson(contour_base_name, region_polygons, args):
    """Writes region polygons as GeoJSON polygons."""
    class_name = args.class_name
    detection_method = args.detection_method
    color = get_color(class_name)
    feature_arr = []
    for polygon in region_polygons:
        properties = {
            "objectType": "annotation",
            "classification": {
                "name": str(class_name),
                "color": list(color) # TODO: color not used in QuPath
            },
            "stained_pixels": polygon.stained_pixels,
            "total_pixels": polygon.total_pixels
        }
        region_coords = polygon.coordinates()
        feature = geojson.Feature(geometry=geojson.Polygon(region_coords), properties=properties)
        feature_arr.append(feature)
    contour_file_name = contour_base_name + '-regions-' + detection_method + '.json'
    with open(contour_file_name, 'w') as geojson_file:
        geojson.dump(feature_arr, geojson_file, indent=2)
    return

def compute_region_polygons(tiles, args):
    min_area = args.min_area
    if ('diagonal' in args.detection_method) and ('_' in args.detection_method):
        factor = float(args.detection_method.split('_')[1])
        min_area = min_area * factor
    class_name = args.class_name
    contour_file_name = args.contour_file
    # Assign region ids to contiguous tiles
    region_ids = compute_regions(tiles, args)
    # Create polygon for each region (ignore -1's)
    unique_region_ids = [x for x in list(set(region_ids)) if x != -1]
    region_polygons = []
    for region_id in unique_region_ids:
        region_tiles = []
        for i in range(len(tiles)):
            if region_ids[i] == region_id:
                region_tiles.append(tiles[i])
        region_polygons.append(compute_polygon_from_tiles(region_tiles))
    # Filter regions by area
    region_polygons_area = []
    for polygon in region_polygons:
        if polygon.area() >= min_area:
            region_polygons_area.append(polygon)
    region_polygons = region_polygons_area
    # If DMN region then filter by proximity to midline
    if class_name == 'dorsal_motor_nucleus':
        region_polygons = filter_dmn_regions(contour_file_name, region_polygons)
    return region_polygons

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contour_file', dest='contour_file', type=str, required=True) 
    parser.add_argument('--tiles_dir', dest='tiles_dir', type=str, required=True)
    parser.add_argument('--class', dest='class_name', choices=CLASS_NAMES, type=str, required=True)
    parser.add_argument('--probability', dest='min_prob', default=0.5, type=float)
    parser.add_argument('--area', dest='min_area', default=0.0, type=float)
    parser.add_argument('--method', dest='detection_method', choices=REGION_DETECTION_METHODS, default='diagonal', type=str)
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_arguments()
    contour_file_name = args.contour_file
    tiles_dir = args.tiles_dir
    class_name = args.class_name
    min_prob = args.min_prob
    detection_method = args.detection_method
    contour_base_name = os.path.splitext(contour_file_name)[0]
    print(f'Detecting regions in contour {contour_file_name} using method {detection_method}')
    # Read tiles: tile format is [x,y,width,height,stain,probability]
    tiles = read_tiles(tiles_dir, class_name)
    print('Read ' + str(len(tiles)) + ' tiles')
    # Filter tiles by min_prob for methods that need it.
    if detection_method in REGION_DETECTION_METHODS_WITH_THRESHOLD:
        tiles = [tile for tile in tiles if tile[5] >= min_prob]
    # Find region polygons
    region_polygons = compute_region_polygons(tiles, args)
    # Write regions to a geojson file
    write_regions_as_geojson(contour_base_name, region_polygons, args)
    print('Found ' + str(len(region_polygons)) + ' regions')
    return

if __name__ == "__main__":
    main()
    