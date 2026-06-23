#uv run training --config configs/pointnet2_nd.yaml
uv run training --config configs/minipointnet_n1.yaml
uv run python eval_benchmark.py --run /home/nrieger/radiance-field-project/runs/minipointnet_2dgs_n_full_0.05_l1_maxp_30 --train 4 --vis-dir vis/ 2>&1 | tee minipointnet_2dgs_n_full_0.05_l1_maxp_30.log
uv run python eval_benchmark.py --run /home/nrieger/radiance-field-project/runs/minipointnet_2dgs_n_full_0.1_maxp_30 --train 4 --vis-dir vis/ 2>&1 | tee minipointnet_2dgs_n_full_0.1_maxp_30.log
uv run python eval_benchmark.py --run /home/nrieger/radiance-field-project/runs/minipointnet_2dgs_n_full_0.01_maxp_30 --train 4 --vis-dir vis/ 2>&1 | tee minipointnet_2dgs_n_full_0.01_maxp_30.log
uv run python eval_benchmark.py --run /home/nrieger/radiance-field-project/runs/minipointnet_2dgs_n_full_0.05_maxp_30 --train 4 --vis-dir vis/ 2>&1 | tee minipointnet_2dgs_n_full_0.05_maxp_30.log
uv run python eval_benchmark.py --run /home/nrieger/radiance-field-project/runs/minipointnet_2dgs_n_full_0.0_maxp_30 --train 4 --vis-dir vis/ 2>&1 | tee minipointnet_2dgs_n_full_0.0_maxp_30.log
#uv run training --config configs/minipointnet_n2.yaml
#uv run training --config configs/linprim.yaml
#uv run training --config configs/pointnet2_n.yaml
#uv run training --config configs/minipointnet_n.yaml
#uv run training --config configs/pointnet2_d.yaml
#uv run training --config configs/minipointnet_d.yaml