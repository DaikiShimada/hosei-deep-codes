#!/bin/sh

curl http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz --output "caltech101.tar.gz"
mkdir caltech101 && tar zxvf caltech101.tar.gz -C caltech101 --strip-components 1

rm -rf caltech101/BACKGROUND_Google
cd caltech101 && find */*.jpg > images_list.csv
