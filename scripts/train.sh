exp_name="ace"

# python main.py --exp_name $exp_name --mode "preprocess"
# python main.py --exp_name $exp_name --mode "merge"
CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "train" --load True
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "evaluate"
# python main.py --exp_name $exp_name --mode "statistic"
# python main.py --exp_name $exp_name --mode "indicator" --sub_mode "evaluate"
# python main.py --exp_name $exp_name --mode "rank" --sub_mode "evaluate"
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "meta"
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "build"