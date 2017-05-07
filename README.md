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