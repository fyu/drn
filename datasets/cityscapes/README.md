## Prepare Cityscapes training data

### Step 1

After you get a vanilla version of Cityscape data label maps, first convert the original segmentation label ids to one of 19 training ids:

```
python3 datasets/cityscapes/prepare_data.py <cityscape folder>/gtFine/
```

### Step 2

- Run `create_lists.sh` in cityscape data folder, containing `gtFine` and `leftImg8bit` to create image and label lists.
- Move [info.json](info.json) to the data folder.