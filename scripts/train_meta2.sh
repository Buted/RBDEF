exp_name="ace_meta2"

# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "train" --load True
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "evaluate" --sub_mode "best-seed42-scale"
# python main.py --exp_name $exp_name --mode "save" --sub_mode "meta20-sample-ent&ev-gamma0.2-gate_2emb-lr9.09e-4"
# python main.py --exp_name $exp_name --mode "save" --sub_mode "44"
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "indicator" --sub_mode "best-seed2021-scale"
python main.py --exp_name $exp_name --mode "rank" --sub_mode "test-base2021-scale"
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "P-R" --sub_mode "best-meta"
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "number"
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "meta"  
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "important"  