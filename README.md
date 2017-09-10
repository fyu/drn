## Image Classification

Go to [this page](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to prepare ImageNet 1K data.

To test a model on ImageNet validation set:
```
python3 classify.py test --arch drn26 -j 4 <imagenet dir> --pretrained
```

To train a new model:
```
python3 classify.py train --arch drn26 -j 8 <imagenet dir> --epochs 120
```

Besides drn26, we also provide drn42 and drn58. They are in DRN-C family as described in [Dilated Residual Networks](https://umich.app.box.com/v/drn).

To cite

```
@inproceedings{Yu2017,
	author    = {Fisher Yu and Vladlen Koltun and Thomas Funkhouser},
	title     = {Dilated Residual Networks},
	booktitle = {CVPR},
	year      = {2017},
}
```

## Semantic Image Segmentataion

### Prepare data

The segmentation image data folder is supposed to contain following image lists with names below:

* train_images.txt
* train_labels.txt
* val_images.txt
* val_labels.txt
* test_images.txt

Each line in the list is a path to an input image or its label map relative to the segmentation folder.

For example, if the data folder is "/foo/bar" and train_images.txt in it contains
```
leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png
```
and train_labels.txt contrains
```
gtFine/train/aachen/aachen_000000_000019_gtFine_trainIds.png
gtFine/train/aachen/aachen_000001_000019_gtFine_trainIds.png
```
Then the first image path is expected at
```
/foo/bar/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
```
and its label map is at
```
/foo/bar/gtFine/train/aachen/aachen_000000_000019_gtFine_trainIds.png
```

In training phase, both train_* and val_* are assumed to be in the data folder. In validation phase, only val_images.txt and val_labels.txt are needed. In testing phase, when there are no available labels, only test_images.txt is needed. `seg.py` has a command line option `--phase` and the corresponding acceptable arguments are `train`, `val`, and `test`.

### Testing on images

Evaluate models on testing set or any images without ground truth labels
```
python3 -u seg.py test -d <data_folder> -c <category_number> --arch drn_d_22
--resume <model_path> --phase test --batch-size 1
```

`category_number` is the number of categories in segmentation. It is 19 for Cityscapes and 11 for Camvid. The actual label maps should contain values in the range of `[0, category_number)`.
