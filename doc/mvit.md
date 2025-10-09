# Multiscale Vision Transformer


## Run

```bash

# dems
WANDB_NAME=cifar-dems sbatch --gpus-per-node=2 hpc/svitrun.sh eval_cls.py  --data_set CIFAR10 --data_location ../data --input_size 32  --ckpt_freq 10  --warmup_epochs=10 --epochs 800 --batch_size=512  --model dems_tiny_patch2_32  --weight_decay 0.05  --gin "build_transform.scale=(0.2,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" build_model.drop_path_rate=0.15 build_model.attn_drop=0.05  --output_dir ../outputs/mvit/cifar_dems_tiny
 

# mvit

WANDB_NAME=cifar-mvit sbatch --gpus-per-node=2 hpc/svitrun.sh eval_cls.py  --data_set CIFAR10 --data_location ../data --input_size 32  --ckpt_freq 10  --warmup_epochs=10 --epochs 800 --batch_size=512  --model mvit2_tiny_patch2  --weight_decay 0.05  --gin "build_transform.scale=(0.2,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)"  --output_dir ../outputs/mvit/cifar_mvit_tiny
```