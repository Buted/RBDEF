exp_name="ace_tmp"

# python main.py --exp_name $exp_name --mode "preprocess"
# python main.py --exp_name $exp_name --mode "merge"
CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "train" --load True
# python main.py --exp_name $exp_name --mode "statistic"
# python main.py --exp_name $exp_name --mode "indicator"
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "meta"