# Deep-CWD-IHC

Deep learning framework for automated Chronic Wasting Disease (CWD) Immunohistochemistry (IHC).

The overall goal of this framework is to identify slides with positive indications of CWD. However, the
framework could also be used to look for similar properties/diseases in other image/tissue types. The
process involves:

1. Isolating the tissue in the images
2. Identifying the type of tissue (lymph node or obex)
3. Finding structures within the tissue (dorsal motor nuclei and follicles)
4. Detecting staining in the tissue structures

Step (2) uses a deep learning model that has been trained on a
manually-prepared set of tissue types: lymph node and obex. Step (3) uses a deep learning model trained
on slides that have been manually annotated with regions denoting positive and negative examples of dorsal
motor nuclei (in obex tissue), and positive and negative examples of follicles (in lymph node tissue).

The image slides are assumed to be in SVS format. The annotations are assumed to be added via QuPath,
which are extracted in GeoJSON format.

## Overview

* `README.md`: This file.
* `environment.yml`: Conda environment.
* `train/`: Scripts used to train DL models.
* `analyze/`: Scripts used to analyze new images.
* `common/`: Scripts used by `analyze` and `train`.
* `utils/`: Scripts use for extracting annotations and annotating images.

The framework is designed as a set of scripts that are called from the command line.
All scripts have detailed comments at the top of the file describing their usage.

## Setup

The Deep-CWD-IHC framework was designed for a conda environment running on a Linux-based system.
The specific environment used for development was Ubuntu 22 running Python 3.11 and Tensorflow 2.15.
A conda environment export is given in `environment.yml`. To setup a similar environment called
`deep-cwd-ihc`, install conda and execute:

`conda env create -f environment.yml`

## Train models

The Deep-CWD-IHC framework relies on three deep neural network models: tissue model (node vs. obex),
node model (follicle vs. non-follicle), and obex model (dorsal motor nucleus (DMV) vs. non-DMV).
Each model accepts a 300x300 pixel tile and outputs the classification of that tile. Each
model has the same structure and is trained in the same way, but using different training tiles.
The following sections describe the processes for generating training tiles and training the models.

### Exporting image annotations

The first step to train models is to identify regions of interest in the slide images. We assume
the use of QuPath for manually annotating images in SVS format. Once the images are annotated, these
annotations can be exported from QuPath using the `utils/export-annotations.sh` shell script. See
details in the script's comments for proper setup. Executing the script will generate an
`<image>.annotations.json` file for each image in the QuPath project. The file describes the
locations of the annotations in GeoJSON format.

`./export-annotations.sh`

### Segmenting images

Next, we need to extract the tissue contours from the images. The `common/segment.py` script is
used for this purpose. The script identifies each tissue contour in the image and outputs a
GeoJSON file describing the points outlining the contour. Each contour is numbered 1, 2, etc.

`python segment.py --image <svs_image_file> --enhance`

The `--enhance` argument can be used to increase the contrast of a low-contrast image. However,
we generally found that this option is not needed.

### Contour tissue labeling

Each of the generated contours must be labeled as either `node` or `obex` for eventual training
of the tissue model. The `train/add_classification.py` script is used for this purpose. For each
contour file geenrated in the previous step, call `add_classification` with the GeoJSON contour
file name and tissue class label for that contour `node` or `obex`. You can optionally provide
a color name (e.g., red, green, blue) to help distinguish the classes when generating visuals
(see description of `utils/annotate.py` later).

`add-classification.py --contour <contour_file> --class <class> [--color <color>]`

### Generate tissue training tiles

Now we are ready to generate the tiles for training the tissue model using the
`common/generate_tiles.py` script. This script is designed to run repeatedly over the
image contours, appending tile images into a tiles directory and appending tile
information to a single CSV file. Below is the command syntax. See the comment at the
top of the script for details.

```
generate-tiles.py --image <svs_slide_file> --annotations <annotations_file>
                  --tiles_dir <tiles_dir> [--overlap <N.N>] [--tile_size <N>]
                  [--tile_increment <N>] [--enhance]
```

The simplest invocation of this script is to just provide the image, the annotations file
(the GeoJSON file for one of the image's tissue contours), and the tiles directory
where to store this information. Call this script repeatedly for each contour of each image
to accumulate the tile information into one tiles directory.

### Collect tissue training tiles

The last step before training the model is to collect the training tiles into a single
directory. The `train/collect-tiles.py` script takes the tiles directory, the destination
training directory

```collect-tiles.py --tiles_dir <tile_dir> --train_dir <train_dir>
                    --class_names class_names --class_column <class_column>
                    [--sample_rate <N.N>]
```

-----------------------------------

### Step 0: Scale slides (optional)

`scale-svs.py [--scale_factor 64] <slide.svs>`

Scales SVS slide to a scaled down PNG image according to the optional scale factor (default = 64).
This step is optional, but affords an easier way to view slides outside QuPath.

### Step 1: Segment tissue

`segment.py <slide.svs>`

Segments the scaled image into tissue and non-tissue components. Identifies contours in the image
that outline major tissue components. For each tissue component, generates an image with that
component against a black background (used later for tissue type detection) and generates a
GeoJSON file with the coordinates of the contour that can be loaded into QuPath using
`load_contour.groovy`. Also generates a copy of the original image with all tissue contours
highlighted in different colors.

Segmentation is based on the marching squares algorithm, which is implemented
in the scikit-image find_contours method.

### Step 2: Detect tissue contour type

`detect-type.py --model_dir <model_dir> [--train <data_dir>] [--test <image_file>]`

First, a DL model is trained to classify the different types of tissue
(e.g., obex, node). After identifying example images of each tissue type,
they are placed in directories like `examples/type1`, `examples/type2`, etc.
Then, the `detect-type.py` script is run with `--train examples` to train
a DL model to classify tissue types. The model information is written to
the `model_dir` directory.

Then, the DL model can be used to classify new contour images by running
the `detect-type.py` script using the `--test` option.

### Step 3: Generate tissue feature tiles

`generate-tiles.py --image <svs_slide> --annotations <annotations> --tiles_dir <tiles_dir>`

Before running `generate-tiles.py` the QuPath annotations for each image need to
be extracted using the `export-annotations.sh` script. See comments in this script
and the accompanying `export-annotations.groovy` script for how to run the script.

Once the annotations are extracted into GeoJSON files, the `generate-tiles.py` script
can be run to generate image tiles that overlap the annotated features in the image.
See the comment at the top of `generate-tiles.py` for more details.

### Step 4: Detect tissue features

`detect-features.py --model_dir <model_dir> [--train <tile_dir>] [--test <tile_file>]`

## Contributors

Lawrence Holder and Liam Broughton-Neiswanger, Washington State University.

## Acknowledgements

The project was supported by grant AP23WSNWRC00C076 from the USDA
Animal and Plant Health Inspection Service (APHIS).
