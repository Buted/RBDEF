exp_name="ace_tmp"

# python main.py --exp_name $exp_name --mode "preprocess"
# python main.py --exp_name $exp_name --mode "merge"
CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "train" --load True 
# python main.py --exp_name $exp_name --mode "statistic"
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "indicator" --sub_mode "evaluate"
# python main.py --exp_name $exp_name --mode "rank" --sub_mode "dev"
# python main.py --exp_name $exp_name --mode "save"
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "meta"
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "threshold"