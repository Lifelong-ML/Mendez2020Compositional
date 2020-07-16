The Birds 200 Dataset
=====================

For more information about the dataset, visit the project website:

  http://www.vision.caltech.edu/visipedia

If you use the dataset in a publication, please cite the dataset in
the style described on the dataset website (see url above).

Directory Information
---------------------

- images/
    The images organized in subdirectories based on species.
- annotations-mat/
    Bounding box and rough segmentation annotations. Organized as
    the images.
- attributes/
    Attribute data from MTurk workers. See README.txt in directory
    for more info.
- attributes-yaml/
    Contains the same attribute data as in 'attributes/' but stored for each
    file as a yaml file with the same name as the image file.
- lists/
    classes.txt : list of categories (species)
    files.txt   : list of all image files (including subdirectories)
    train.txt   : list of all images used for training
    test.txt    : list of all images used for testing
    splits.mat  : training/testing splits in MATLAB .mat format
    