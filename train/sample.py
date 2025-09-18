# sample.py --source <source_dir> --target <target_dir> --rate <sampling_rate>
#
# Copies a random sample of the files in source directory to the
# target directory according to the sampling rate (0.0,1.0).
# The source_dir is assumed to have subdirectories for each class
# and then images within these subdirectories. The sampling rate
# is applied to each class independently. The target_dir will have
# the same directory structure as the source_dir.
#
# Author: Lawrence Holder, Washington State University

import os
import shutil
import argparse
import random

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source_dir', type=str, required=True)
    parser.add_argument('--target', dest='target_dir', type=str, required=True)
    parser.add_argument('--rate', dest='sampling_rate', type=float, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    source_dir = args.source_dir
    target_dir = args.target_dir
    sampling_rate = args.sampling_rate
    # Get source subdirs
    source_subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    # Ensure target directories exist
    os.makedirs(target_dir, exist_ok=True)
    for subdir in source_subdirs:
        subdir_path = os.path.join(target_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
    # Copy sampling of images from each subdir
    num_orig = 0
    num_sample = 0
    for subdir in source_subdirs:
        source_subdir = os.path.join(source_dir, subdir)
        target_subdir = os.path.join(target_dir, subdir)
        source_files = os.listdir(source_subdir)
        source_image_files = [f for f in source_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        sample_size = int(sampling_rate * len(source_image_files))
        target_image_files = random.sample(source_image_files, sample_size)
        print(f'sampled {len(target_image_files)} of {len(source_image_files)} images from {subdir}')
        num_orig += len(source_image_files)
        num_sample += len(target_image_files)
        for file_name in target_image_files:
            source_file_path = os.path.join(source_subdir, file_name)
            shutil.copy2(source_file_path, target_subdir)
    print(f'sampled total {num_sample} of {num_orig} from {source_dir}')
if __name__ == "__main__":
    main()
