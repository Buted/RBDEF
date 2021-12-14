exp_name="ace_meta"

CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "train"
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_name $exp_name --mode "evaluate" --sub_mode "best-dropout0.3-lr5e-3"
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "meta"
# python main.py --exp_name $exp_name --mode "save" --sub_mode "best_acc-fixed1024-lr5e-4"
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_name $exp_name --mode "indicator" --sub_mode "best-seed99"
# python main.py --exp_name $exp_name --mode "rank" --sub_mode "best-seed99-dev"