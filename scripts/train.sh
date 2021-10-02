exp_name="ace"

# python main.py --exp_name $exp_name --mode "preprocess"
# python main.py --exp_name $exp_name --mode "merge"
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name $exp_name --mode "train"
# python main.py --exp_name $exp_name --mode "statistic"
# python main.py --exp_name $exp_name --mode "indicator" --sub_mode "evaluate"
# python main.py --exp_name $exp_name --mode "rank" --sub_mode "evaluate"