exp_name="kbp"

# python main.py --exp_name $exp_name --mode "preprocess"
# python main.py --exp_name $exp_name --mode "merge"
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "train" --load True
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "evaluate" --load True
# python main.py --exp_name $exp_name --mode "statistic"
# python main.py --exp_name $exp_name --mode "indicator" --sub_mode "best-seed84"
# python main.py --exp_name $exp_name --mode "rank" --sub_mode "best-seed84-False"
# python main.py --exp_name $exp_name --mode "save" --sub_mode "best"
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "meta"
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "threshold"
# python main.py --exp_name $exp_name --mode "group"
# python main.py --exp_name $exp_name --mode "divide"
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "important"