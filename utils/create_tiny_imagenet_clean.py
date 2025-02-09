#! /usr/bin/env python
# reformat data to abide by hf formatting for file-based datasets
import os
import shutil
from pathlib import Path
from datasets import load_dataset
import argparse

def cleanup_tiny_imagenet(base_path):
    """
    Move JPEG files from images subdirectory to parent directory in Tiny ImageNet dataset
    
    Args:
        base_path (str): Base path to the Tiny ImageNet dataset
    """
    # Ensure base path exists
    base_path = Path(base_path)
    if not base_path.exists():
        raise ValueError(f"Path does not exist: {base_path}")
    
    # Find all 'images' directories
    image_dirs = list(base_path.rglob('*/images'))
    
    if not image_dirs:
        print(f"No 'images' directories found in {base_path}")
        return
    
    # Track total files moved
    total_moved = 0
    
    # Process each images directory
    for images_path in image_dirs:
        # Parent directory where files will be moved
        class_path = images_path.parent
        
        # Find all JPEG files in images directory
        jpeg_files = list(images_path.glob('*.JPEG'))
        
        if not jpeg_files:
            print(f"No JPEG files found in {images_path}")
            continue
        
        # Move files
        for jpeg_file in jpeg_files:
            destination = class_path / jpeg_file.name
            shutil.move(str(jpeg_file), str(destination))
            total_moved += 1
            print(f"Moved {jpeg_file.name} to {destination}")
        
        # Remove empty images directory
        if not os.listdir(images_path):
            os.rmdir(images_path)
            print(f"Removed empty directory: {images_path}")
    
    print(f"Total files moved: {total_moved}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='../data/image/tiny-imagenet-200',
                        help='Base path to the Tiny ImageNet dataset')
    args = parser.parse_args()
    return args
   
def main():
    # Default path, can be modified as needed
    # http://cs231n.stanford.edu/tiny-imagenet-200.zip
    args = get_args()
    # base_path = '../data/image/tiny-imagenet-200'
    # reformat paths to be readable by HF datasets
    cleanup_tiny_imagenet(args.base_path)
    # load once filter out grey images and save to disk for faster reloading
    ds = load_dataset('imagefolder', data_dir='../data/image/tiny-imagenet-200')
    ds = ds.filter(lambda row: row['image'].mode == 'RGB')
    ds.save_to_disk('../data/image/tiny-imagenet-200-clean')

if __name__ == '__main__':
    main()