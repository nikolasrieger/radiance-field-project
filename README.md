# Radiance Field Project

VERY IMPORTANT NOTE: ALWAYS USE FOREGROUND MASK!!!

## Setup

1. Download the dataset via: ```uv run download-dataset --download_small_subset --download_folder ./data/uco3d_d
ata_raw```.
2. Get the rasterizer: ```git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git```
```cd diff-gaussian-rasterization```
```pip install .```

### Downloading a larger training subset (~10k samples)

For training larger models you can grow the dataset beyond the small debug subset. uCO3D archives are bundled per super-category, so an exact sample count isn't possible — instead, ```download-10k-subset``` reads the dataset ```metadata.sqlite```, greedily selects whole super-categories (smallest-first, to maximize category diversity) until the cumulative sequence count reaches a target (~10000 by default), and downloads the ```rgb_videos``` + ```gaussian_splats``` modalities for those categories. Super-categories already present on disk count toward the target and are not re-downloaded, and ```metadata``` is skipped automatically when ```metadata.sqlite``` already exists.

1. Preview the plan (dry-run by default — prints the selected categories, projected sample count, and estimated download size): ```uv run download-10k-subset```
2. Start the download into ```./data/uco3d_data_big```: ```uv run download-10k-subset --run```

Useful flags: ```--target 15000``` (change the sample target), ```--download_folder <path>```, ```--n_download_workers 8```, ```--checksum_check```, ```--clear_archives_after_unpacking``` (delete zips after extraction to save disk). Note that the metadata (which ships in ```metadata.zip``` / ```metadata.sqlite```) must already be present in the download folder for the script to compute per-category counts.

## Training a model

1. Pick the config of the wanted model from the config folder or create a new config.
2. Run it via: ```uv run training --config configs/minipointnet.yaml```

## Evaluation

1. Evaluate a trained model on different metrics (currently PSNR, cos similarity and run time): ```uv run python eval_benchmark.py --run /radiance-field-project/runs/minipointnet_2dgs_d_full_0.1_maxp_30 --train 4 --vis-dir vis/``` (when omitting train, validation data will be used).
2. View the output of a trained model in an interactivate 3D viewer: ```uv run python view_splats_interactive.py --run /radiance-field-project/runs/minipointnet_2dgs_d_full_0.1_maxp_30```
3. Compute the losses on a validation set: ```uv run python evaluate_splat.py --run /radiance-field-project/runs/minipointnet_2dgs_d_full_0.1_maxp_30```, can be also run for all saved model checkpoint by running: ```uv run eval_progression.py```.

