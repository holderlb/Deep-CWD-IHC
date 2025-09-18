# export-annotations.sh
#
# Shell script to run QuPath from command-line and output the
# annotations for each image in the given project file.
#
# Modify executable path to QuPath according to your platform.
# Set --project to location of your QuPath project file.
# Run this script in the same directory with the export-annotations.groovy
# script.
#
# Author: Lawrence Holder, Washington State University

/Applications/QuPath.app/Contents/MacOS/QuPath script \
  --project=project.qpproj \
  export-annotations.groovy
