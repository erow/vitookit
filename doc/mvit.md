# Multiscale Vision Transformer


## Run

```bash

# vit: 96.94
WANDB_NAME=cifar_r64-vitt sbatch hpc/svitrun.sh eval_cls.py --opt adamw --lr 1e-3 --warmup_epochs=10 --epochs 200 --batch_size=256 --ra=3 --weight_decay 0.05 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 64 --gin build_transform.scale="(0.8,1)" "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.img_size=64 build_model.patch_size=4 --model vit_tiny_patch16_224 --output_dir ../outputs/mvit/cifar64_vit_tiny

# vit: 95.66
WANDB_NAME=cifar-vitt sbatch hpc/svitrun.sh eval_cls.py --opt adamw --lr 1e-3 --warmup_epochs=10 --epochs 200 --batch_size=256 --ra=3 --weight_decay 0.05 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --gin build_transform.scale="(0.8,1)" "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.img_size=32 build_model.patch_size=2 --model vit_tiny_patch16_224 --output_dir ../outputs/mvit/cifar_vit_tiny

# dems: pretrain, official 96.03, ours: 95.8
WANDB_NAME=cifar-dems sbatch --gpus-per-node=2 hpc/svitrun.sh eval_cls.py  --data_set CIFAR10 --data_location ../data --input_size 32  --ckpt_freq 10  --warmup_epochs=10 --epochs 800 --batch_size=512  --model dems_tiny_patch2_32  --weight_decay 0.05  --gin "build_transform.scale=(0.2,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" build_model.drop_path_rate=0.2 build_model.attn_drop=0.05  --output_dir ../outputs/mvit/cifar_dems_tiny

# dems: ft using weak aug, official 96.74
WANDB_NAME=cifar-dems-ft sbatch --gpus-per-node=2 hpc/svitrun.sh eval_cls.py  --data_set CIFAR10 --data_location ../data --input_size 32  --ckpt_freq 5 --blr=1e-3 --warmup_epochs=10 --epochs 100 --aa="" --reprob=0 --disable_weight_decay_on_bias_norm  --batch_size=512  --model dems_tiny_patch2_32  --weight_decay 0.05  --gin "build_transform.scale=(1,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" build_model.drop_path_rate=0.2 build_model.attn_drop=0.05 \
 -w ../outputs/mvit/cifar_dems_tiny/checkpoint.pth --checkpoint_key=model --output_dir ../outputs/mvit/cifar_dems_tiny/ft

# dems: evaluate
vitrun eval.py  --data_set CIFAR10 --data_location ../data --input_size 32   --model dems_tiny_patch2_32   --gin "build_transform.scale=(0.2,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" -w ../outputs/mvit/cifar_dems_tiny/checkpoint.pth --checkpoint_key=model  

# mvit: pretrain: 96.58
WANDB_NAME=cifar-mvit sbatch --gpus-per-node=2 hpc/svitrun.sh eval_cls.py  --data_set CIFAR10 --data_location ../data --input_size 32  --ckpt_freq 10  --warmup_epochs=0 --epochs 800 --batch_size=512  --model mvit2_tiny_patch2  --weight_decay 0.05  --gin "build_transform.scale=(0.2,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)"  --output_dir ../outputs/mvit/cifar_mvit_tiny

# mvit: ft using weak aug:
WANDB_NAME=cifar-mvit-ft sbatch hpc/svitrun.sh eval_cls.py  --data_set CIFAR10 --data_location ../data --input_size 32  --ckpt_freq 5 --blr=1e-3 --warmup_epochs=0 --epochs 100 --aa="" --reprob=0 --disable_weight_decay_on_bias_norm  --batch_size=1024  --model mvit2_tiny_patch2  --weight_decay 0.05  --gin "build_transform.scale=(1,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" build_model.drop_path_rate=0.2 build_model.attn_drop=0.05  -w ../outputs/mvit/cifar_mvit_tiny/checkpoint.pth --checkpoint_key=model --output_dir ../outputs/mvit/cifar_mvit_tiny/ft

```


## augmentation
Strong augmentation for pretraing:
```
Transform: Compose(
               RandomResizedCropAndInterpolation(size=(32, 32), scale=(0.2, 1), ratio=(0.75, 1.3333), interpolation=bicubic)
               RandomHorizontalFlip(p=0.5)
               RandAugment(n=2, ops=
                AugmentOp(name=AutoContrast, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=Equalize, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=Invert, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=Rotate, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=PosterizeIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=SolarizeIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=SolarizeAdd, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ColorIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ContrastIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=BrightnessIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=SharpnessIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ShearX, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ShearY, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=TranslateXRel, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=TranslateYRel, p=0.5, m=9, mstd=0.5))
               MaybeToTensor()
               Normalize(mean=tensor([0.4914, 0.4822, 0.4465]), std=tensor([0.2470, 0.2435, 0.2616]))
               RandomErasing(p=0.25, mode=pixel, count=(1, 1))
           )
```

Weak augmentation for finetuning
```
Transform: Compose(
    RandomResizedCropAndInterpolation(size=(32, 32), scale=(1, 1), ratio=(0.75, 1.3333), interpolation=bicubic)
    RandomHorizontalFlip(p=0.5)
    MaybeToTensor()
    Normalize(mean=tensor([0.4914, 0.4822, 0.4465]), std=tensor([0.2470, 0.2435, 0.2616]))
)
```