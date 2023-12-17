# The CropAndWeed Dataset
A thorough description can be found in the corresponding paper and supplementary material, which will be published at WACV 2023. Please see https://github.com/cropandweed/cropandweed-dataset for more information.
The dataset is distributed using five files: cropandweed_annotations.tar containing all annotations and meta data; cropandweed_images1of4.tar, cropandweed_images1of4.tar, cropandweed_images1of4.tar, and cropandweed_images4of4.tar containing the intensity images.

## Annotation Format
The annotations consist of multiple directories for each dataset variant in the following formats:
* `bboxes` contains csv-files for each image with object instances defined as: `Left, Top, Right, Bottom, Label ID, Stem X, Stem Y`
* `labelIds` contains semantic masks for each image
* `params` contains the following additional parameters for each image:
  * `moisture`: 0 (dry), 1 (medium) or 2 (wet)
  * `soil`: 0 (fine), 1 (medium) or 2 (coarse)
  * `lighting`: 0 (sunny) or 1 (diffuse)
  * `separability`: 0 (easy), 1 (medium), 2 (hard)

The corresponding label IDs for each datset variant are specified in [datasets.py](cnw/utilities/datasets.py). 
The names of all image and annotation files are prefixed either with _ave_ or _vwg_ refering to the Application and Experimental Sets, respectively, as described in the paper. 
The following 4-digit numbers specify the recording session, while the last 4 digits are the image id.  

## Licence
The CropAndWeed dataset is released to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications or personal experimentation ([LICENCE](LICENCE)).

## Citing
If you use the CropAndWeed dataset for your research, please use the following BibTeX entry:

```BibTeX
@InProceedings{Steininger2023CropAndWeedDataset,
    author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
    title     = {The CropAndWeed Dataset: a Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {tbd-tbd}
}
```
