# export-annotations.sh
#
# Shell script to run QuPath from command-line and output the
# annotations for each image in the given project file.
#
# Set path to QuPath according to your platform.
#
# Author: Lawrence Holder, Washington State University

/Applications/QuPath.app/Contents/MacOS/QuPath script \
  --project=/Users/holder/projects/usda/qupath/Annotations/10-23-23/project.qpproj \
  export-annotations.groovy
