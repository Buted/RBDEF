exp_name="ace_meta"

CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "train" 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "meta"
# python main.py --exp_name $exp_name --mode "save"
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_name $exp_name --mode "indicator" --sub_mode "evaluate"
# python main.py --exp_name $exp_name --mode "rank" --sub_mode "dev"