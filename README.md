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
Each model accepts a 300x300 pixel tile and outputs the classification of that tile. Each
model has the same structure and is trained in the same way, but using different training tiles.

The image slides are assumed to be in SVS format. The annotations are assumed to be added via QuPath,
which are extracted in GeoJSON format.

## Overview

* `README.md`: This file.
* `environment.yml`: Conda environment.
* `train/`: Scripts used to train DL models.
* `analyze/`: Scripts used to analyze new images.
* `common/`: Scripts used by `analyze` and `train`.
* `models/`: Where trained models will reside.
* `utils/`: Scripts use for extracting annotations and annotating images.

The framework is designed as a set of scripts that are called from the command line.
All scripts have detailed comments at the top of the file describing their usage.

## Setup

The Deep-CWD-IHC framework was designed for a conda environment running on a Linux-based system.
The specific environment used for development was Ubuntu 22 running Python 3.11 and Tensorflow 2.15.
A conda environment export is given in `environment.yml`. To setup a similar environment called
`deep-cwd-ihc`, install conda and execute:

`conda env create -f environment.yml`

## Train tissue model (node vs. obex)

Below are the steps for training the tissue model that classifies a tile as belonging to either
node or obex tissue.

### Segment images (find tissue contours)

First, we need to extract the tissue contours from the images. The `common/segment.py` script is
used for this purpose. The script identifies each tissue contour in the image and outputs a
GeoJSON file describing the points outlining the contour. Each contour is numbered 1, 2, etc.

`segment.py --image <svs_image_file>`

### Label tissue contours

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
                  [--tile_increment <N>]
```

The simplest invocation of this script is to just provide the image, the annotations file
(the GeoJSON file for one of the image's tissue contours), and the tiles directory
where to store this information. Call this script repeatedly for each contour of each image
to accumulate the tile information into one tiles directory.

### Collect tissue training tiles

The last step before training the model is to collect the training tiles into a single
directory. The `train/collect-tiles.py` script takes the tiles directory, the destination
training directory, and the class names to collect (i.e., `node` and `obex`).

```collect-tiles.py --tiles_dir <tile_dir> --train_dir <train_dir>
                    --class_names class_names [--class_column <class_column>]
                    [--sample_rate <N.N>]
```

The class column can be omitted assuming you are using the same column name `directory`
that was used by `generate_tiles`. The default sampling rate is 1.0, but you can choose
a smaller value to reduce the number of actual tiles used for training. The sampling
rate is applied to each class independently.

### Train tissue model

Finally, we are ready to train the tissue model using the `train/train-model.py` script.
The simplest invocation is to provide the training directory from the previous step.
The model will be trained using these tiles. The trained model is written in Keras
format. See the comment at the top of the script for details on the other arguments.

```
train-model.py --data_dir <data_dir> [--model_file <model_file>]
              [--pretrained_model_file <pre_model_file>]
              [--batch-size 32] [--epochs 300] [--unfreeze 20]
```

This model, which we refer to as `model-tissue.keras` will be used later to analyze
new images.

## Train node and obex models (DMN vs. non-DMN, follicle vs. non-follicle)

Below are the steps for training the two structure prediction models: the obex model that
predicts if a tile overlaps a DMN or not, and the node model that predicts if a tile
overlaps a follicle or not.

### Exporting image annotations

The first step to train the models is to identify regions of interest in the slide images. We assume
the use of QuPath for manually annotating images in SVS format. In our case, we used the annotations:
`dorsal_motor_nucleus`, `not_dmn`, `follicle`, `non-follicular`.

Once the images are annotated, these
annotations can be exported from QuPath using the `utils/export-annotations.sh` shell script. See
details in the script's comments for proper setup. Executing the script will generate an
`<image>.annotations.json` file for each image in the QuPath project. The file describes the
locations of the annotations in GeoJSON format.

`./export-annotations.sh`

### Generate node/obex training tiles (and staining information)

We generate training tiles for using the same
`common/generate_tiles.py` script as used above, except we will use the GeoJSON annotations from
the previous step. This script is designed to run repeatedly over the
training images, appending tile images into a tiles directory and appending tile
information to a single CSV file. Below is the command syntax. See the comment at the
top of the script for details.

```
generate-tiles.py --image <svs_slide_file> --annotations <annotations_file>
                  --tiles_dir <tiles_dir> [--overlap <N.N>] [--tile_size <N>]
                  [--tile_increment <N>]
```

The simplest invocation of this script is to just provide the image, the annotations file
(the GeoJSON file for one of the image's tissue contours), and the tiles directory
where to store this information. Call this script repeatedly for each image
to accumulate the tile information into one tiles directory.

The `generate_tiles` script also analyzes tiles for the presence of staining. A `stained_pixels`
column is added to the CSV file, whose value is the number of pixels in the 300x300 tile
containing stain. This information is used later in the analysis phase.

### Collect node/obex training tiles

The `train/collect-tiles.py` script is again used to collect tiles of each type
into from the tiles directory to a destination training directory with sub-directories
for each class name: `dorsal_motor_nucleus` and `not_dmn` for the obex model,
`follicle` and `non-follicular` for the node model. So, `collect_tiles` will be
called twice: once for collecting node model training tiles, and once for collecting
obex model training tiles.

```
collect-tiles.py --tiles_dir <tile_dir> --train_dir <train_dir>
                    --class_names class_names [--class_column <class_column>]
                    [--sample_rate <N.N>]
```

The class column can be omitted assuming you are using the same column name `directory`
that was used by `generate_tiles`. The default sampling rate is 1.0, but you can choose
a smaller value to reduce the number of actual tiles used for training. The sampling
rate is applied to each class independently.

### Train node/obex models

Finally, we are ready to train the node and obex models using the `train/train-model.py`
script. The simplest invocation is to provide the training directory from the previous step.
The model will be trained using these tiles. The trained model is written in Keras
format. See the comment at the top of the script for details on the other arguments.

```
train-model.py --data_dir <data_dir> [--model_file <model_file>]
              [--pretrained_model_file <pre_model_file>]
              [--batch-size 32] [--epochs 300] [--unfreeze 20]
```

This script will be run twice: once to generate the node model, which we refer to as
`model-node.keras`, and once to generate the obex model, which we refer to as
`model-obex.keras`.

## Analyze new image

Below are the steps for analyzing a new image.

### Setup

Before analyzing a new image, the three models
(`model-tissue.keras`, `model-node.keras`, `model-obex.keras`) should be copied into
the `models` directory within the main repository directory. Accompanying each model
should be a JSON file describing the class names predicted by the model, in the proper
order. The first name should correspond to the model's output 0, and the second name should
correspond to the model's output 1. Assuming the training process above is followed, the
JSON files include in the repository will suffice.

The analysis script requires the input of a metadata JSON file setting various
parameters for the analysis process. A sample `analyze/metadata.json`
file has been provided with values that have been found to yield
good results, but you can tweak these parameters if desired.

### Analysis

To analyze a new image, we use the `analyze/analyze.py` script.

`analyze.py --image_file <image_file.svs> --metadata <metadata_file>`

This script calls all the other scripts to analyze the image. The process is long and
complicated, so we do not detail it here. See the calls at the bottom of the `analyze.py`
script for details.

The script creates an HTML directory `<image_file>-html` containing the results of the
analysis. Load the `index.html` file from this directory to view the results.

## Visualization tool

The `utils/annotate.py` script is provided to support various visualizations of the
above processes.

```
annotate.py --image <svs_slide> --geojson <geojson_file>
            --errors <color> --class <class> --probability <prob>
            --tiles_dir <tiles_dir> 
```

The script scales and annotates the given image with contours provided by the
GeoJSON file and tiles found in the tiles directory. Displayed tiles can be
filtered by a particular class and a particular prediction probability threshold.
Tiles incorrectly predicted can be highlighted in a different color.
See the comment at the top of the script for more details.

## Contributors

Lawrence Holder and Liam Broughton-Neiswanger, Washington State University.

## Acknowledgements

The project was supported by grant AP23WSNWRC00C076 from the USDA
Animal and Plant Health Inspection Service (APHIS).
