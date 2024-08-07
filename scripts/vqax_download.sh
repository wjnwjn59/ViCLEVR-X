#!/bin/bash

# Define the dataset directory variable
dataset_dir="datasets"

# Check if the dataset directory exists
if [ ! -d "$dataset_dir" ]; then
  mkdir -p "$dataset_dir"
  echo "Created directory: $dataset_dir"
fi

# Function to download and extract a dataset
download_and_extract() {
  dataset_name="$1"
  dataset_url="$2"
  archive_name="$dataset_name.zip"

  # Check if dataset directory exists
  if [ -d "$dataset_dir/$dataset_name" ] && [ "$(ls -A "$dataset_dir/$dataset_name")" ]; then
    echo "$dataset_name dataset already exists."
  else
    # Download the dataset
    echo "Downloading $dataset_name dataset..."
    wget "$dataset_url" -O "$dataset_dir/$archive_name"

    # Extract the dataset
    echo "Extracting $dataset_name dataset..."
    unzip "$dataset_dir/$archive_name" -d "$dataset_dir/$dataset_name"
    rm "$dataset_dir/$archive_name"
  fi
}

# Download VQAv2 training images
download_and_extract "vqa_v2_training" "http://images.cocodataset.org/zips/train2014.zip"

# Download VQAv2 validation images
download_and_extract "vqa_v2_val" "http://images.cocodataset.org/zips/val2014.zip"

# Download VQAv2 testing images
download_and_extract "vqa_v2_test" "http://images.cocodataset.org/zips/test2015.zip"

# Download the VQA-X train annotations
wget --id 1b3x4ku3LlOGEoFPQiVjDscIy4Spn-UGJ -O ./datasets

# Download the VQA-X val annotations
wget --id 1PgGqu8R9tzWReJW4epsLZ5VNXIE_dxW_ -O ./datasets

# Download the VQA-X test annotations
wget --id 1P3hSie3osvy2_tcjgMrgy4ZulkkXktbt -O ./datasets

# # Download CLEVR-X dataset (replace with the actual URL)
# download_and_extract "CLEVR-X" "https://www.dropbox.com/sh/qe1wfahldk3pd7l/AADnsGTUInU5-eLCjyor0Iapa?dl=1"

echo "Download and extraction complete."
