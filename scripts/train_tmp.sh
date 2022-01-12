exp_name="ace_tmp"

# python main.py --exp_name $exp_name --mode "preprocess"
# python main.py --exp_name $exp_name --mode "merge"
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "train" --load True
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "evaluate" --sub_mode "best-seed519-scale"
# python main.py --exp_name $exp_name --mode "statistic"
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "indicator" --sub_mode "best-seed1024-scale"
python main.py --exp_name $exp_name --mode "rank" --sub_mode "test-base1024-scale"
# python main.py --exp_name $exp_name --mode "save" --sub_mode "21"
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "meta"  
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "threshold"
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "P-R" --sub_mode "best"