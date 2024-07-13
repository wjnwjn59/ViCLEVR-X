sudo apt-get install wget
sudo apt-get install unzip

#!/bin/bash
mkdir -p dataset

# Download ViCLEVR dataset
wget -O dataset/data.zip "https://www.dropbox.com/sh/qe1wfahldk3pd7l/AADnsGTUInU5-eLCjyor0Iapa?dl=1"

unzip dataset/data.zip -d dataset
rm dataset/data.zip

echo "Download and extraction complete."