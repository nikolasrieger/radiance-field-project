# Radiance Field Project

## Setup

1. Download the dataset via: ```uv run download-dataset --download_small_subset --download_folder ./data/uco3d_d
ata_raw```.
2. Get the rasterizer: ```git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git```
```cd diff-gaussian-rasterization```
```pip install .```

## Training a model

1. Pick the config of the wanted model from the config folder or create a new config.
2. Run it via: ```uv run training --config configs/minipointnet.yaml```

## Evaluation

1. Evaluate a trained model on different metrics (currently PSNR, cos similarity and run time): ```uv run python eval_benchmark.py --run /radiance-field-project/runs/minipointnet_2dgs_d_full_0.1_maxp_30 --train 4 --vis-dir vis/``` (when omitting train, validation data will be used).
2. View the output of a trained model in an interactivate 3D viewer: ```uv run python view_splats_interactive.py --run /radiance-field-project/runs/minipointnet_2dgs_d_full_0.1_maxp_30```
3. Compute the losses on a validation set: ```uv run python evaluate_splat.py --run /radiance-field-project/runs/minipointnet_2dgs_d_full_0.1_maxp_30```, can be also run for all saved model checkpoint by running: ```uv run eval_progression.py```.

