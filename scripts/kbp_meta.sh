exp_name="kbp_meta"

CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "train" --load True
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "evaluate" --sub_mode "best-dropout0.3-lr5e-3"
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "meta"
# python main.py --exp_name $exp_name --mode "save" --sub_mode "meta"
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "indicator" --sub_mode "best-seed99"
# python main.py --exp_name $exp_name --mode "rank" --sub_mode "best-seed99-False"
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "threshold"