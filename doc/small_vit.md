# Small Vision Transformer


## CIFAR10
```bash
WANDB_NAME=cifar-vitt sbatch hpc/svitrun.sh eval_cls.py --opt adamw --lr 1e-3 --warmup_epochs=10 --epochs 200 --batch_size=256 --ra=3 --weight_decay 0.05 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --gin build_transform.scale="(0.8,1)" "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.img_size=32 build_model.patch_size=2 --model vit_tiny_patch16_224 --output_dir ../outputs/mvit/cifar_vit_tiny


# cait
WANDB_NAME=cifar-cait sbatch  svitrun.sh eval_cls.py --opt adamw --lr 1e-3 --warmup_epochs=10 --epochs 200 --batch_size=256 --ra=3 --weight_decay 0.05 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --gin build_transform.scale="(0.8,1)" "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.img_size=32 build_model.patch_size=2 --model cait_xxt12_32 --output_dir ../outputs/mvit/cifar_cait_tiny

WANDB_NAME=cifar-cait_c3 sbatch  svitrun.sh eval_cls.py --opt adamw --lr 1e-3 --warmup_epochs=10 --epochs 200 --batch_size=256 --ra=3 --weight_decay 0.05 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --gin build_transform.scale="(0.8,1)" "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.img_size=32 build_model.patch_size=2 build_model.depth_token_only=3 --model cait_xxt12_32 --output_dir ../outputs/mvit/cifar_cait_c3_tiny

# filter vit
WANDB_NAME=cifar-fit sbatch  svitrun.sh eval_cls.py --opt adamw --lr 1e-3 --warmup_epochs=10 --epochs 200 --batch_size=256 --ra=3 --weight_decay 0.05 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --gin build_transform.scale="(0.8,1)" "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.img_size=32 build_model.patch_size=2 --model fit_s12c2_32 --output_dir ../outputs/mvit/cifar_fit_tiny

# think vit
WANDB_NAME=cifar-thinkvit sbatch  svitrun.sh eval_cls.py --opt adamw --lr 1e-3 --warmup_epochs=10 --epochs 200 --batch_size=256 --ra=3 --weight_decay 0.05 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --gin build_transform.scale="(0.8,1)" "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.img_size=32 build_model.patch_size=2 --model thinkvit_s12c4_32  --output_dir ../outputs/mvit/cifar_thinkvit_tiny
```

