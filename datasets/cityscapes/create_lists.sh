#!/usr/bin/env bash

find leftImg8bit/train -maxdepth 3 -name "*_leftImg8bit.png" | sort > train_images.txt
find leftImg8bit/val -maxdepth 3 -name "*_leftImg8bit.png" | sort > val_images.txt
find leftImg8bit/test -maxdepth 3 -name "*_leftImg8bit.png" | sort > test_images.txt

find gtFine/train -maxdepth 3 -name "*_trainIds.png" | sort > train_labels.txt
find gtFine/val -maxdepth 3 -name "*_trainIds.png" | sort > val_labels.txt

