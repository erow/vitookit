# Image Classification

We provide a flexible configuration system for training and evaluating models. You can use the `--gin` flag to pass in configuration options. In this guidline, we will show you some training recipes for resnet50 and vision transformers.

## ResNet

Our default recipe for training resnet50 on ImageNet1K is as follows:

```bash
# our recipe
name=Random-IN1K 
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --blr 5e-4 --weight_decay 2e-5  --gin  build_model.model_name="'resnet50'" --output_dir outputs/cls/${name} 


# Ross' recipy: https://arxiv.org/pdf/2110.00476#page=5.70
name=Random_ross-IN1K 
WANDB_NAME=$name  vitrun --nproc_per_node 8 timm_train.py ~/data/ImageNet -b 64 --model resnet50 --sched cosine --epochs 200 --blr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce --log-wandb  --output outputs/cls/${name}

```

With pretrained weights:
```bash

name=Random-IN1K 
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --opt LAMB --blr 1e-3 --weight_decay 2e-2 --smoothing=0 --layer_decay 0.75 --gin  build_model.model_name="'resnet50'" --output_dir outputs/cls/${name} 



# Ross' recipy: https://arxiv.org/pdf/2110.00476#page=5.70
name=Random_ross-IN1K 
WANDB_NAME=$name  vitrun --nproc_per_node 8 timm_train.py ~/data/ImageNet -b 64 --model resnet50 --sched cosine --epochs 200 --blr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce   --pretrained-path ${weight} --pretrained  --layer-decay 0.75 --log-wandb  --output outputs/cls/${name}

```

## Vision transformer


```bash
name=vitb-default-IN1K 
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=128  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --opt AdamW --opt_betas 0.9 .95 --blr 5e-4 --epochs 300 --weight_decay 5e-2 --smoothing=0.1 --reprob 0.1  --gin   build_model.drop_path_rate=0.1 --output_dir outputs/cls/${name} 
# 81.2642415364



# MAE recipe without EMA (82.1% w/o EMA, 82.3% w EMA): https://arxiv.org/pdf/2111.06377#page=12
name=vitb-default-IN1K 
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=512  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --opt adamw --opt_betas 0.9 .95 --blr 1e-4 --weight_decay 0.3 --smoothing=0.1 --warmup=20 --epochs=300  --gin   build_model.drop_path_rate=0.1 --output_dir outputs/cls/${name} 

# Lion recipe : https://arxiv.org/pdf/2302.06675#page=29.21
name=vitb-lion-IN1K 
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --opt lion --blr 1e-5 --weight_decay 0.3 --warmup=20 --epochs 300  --smoothing=0.1 --reprob 0.1  --gin   build_model.drop_path_rate=0.1 --output_dir outputs/cls/${name} 


name=vitb-lion-IN1K-wd2
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --opt lion --blr 1e-5 --weight_decay 2 --warmup=20 --epochs 300  --smoothing=0.1 --reprob 0.1  --gin   build_model.drop_path_rate=0.1 --output_dir outputs/cls/${name} 
```

```bash
name=vitb-iNat18
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=128  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --opt AdamW --opt_betas 0.9 .95 --blr 1e-5  --epochs 300 --weight_decay 5e-2 --smoothing=0.1 --reprob 0.1 --gin  build_model.drop_path_rate=0.1 \
-w <weights> --output_dir outputs/cls/${name} 
```  

```bash
name=vitb-IN1K
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=128  --train_path $FFCVTRAIN --val_path $FFCVVAL  --ckpt_freq 5 --opt AdamW --opt_betas 0.9 .95 --blr 1e-4  --epochs 100 --weight_decay 5e-2 --smoothing=0.1 --reprob 0.1 --gin  build_model.drop_path_rate=0.1 \
-w <weights> --output_dir outputs/cls/${name} 
```

From pretrained weights:
```bash
name=MAE_base-IN1K
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls.py --batch_size=128 --data_location $IMNET  --ckpt_freq 5 --opt adamw --opt_betas 0.9 .999 --blr 5e-4 --reprob 0.25 --mixup 0.8 --cutmix 1.  --weight_decay 0.05 --layer_decay=0.65 --smoothing=0.1 --warmup_epochs=5 --epochs=100  --gin   build_model.drop_path_rate=0.1 --output_dir outputs/cls/${name} --checkpoint_key model -w ../models/vitb/mae_pretrain_vit_base.pth 
#83.53 https://wandb.ai/erow/vitookit/runs/koswtcoj?nw=nwusererow

name=MAE_base-IN1K-ffcv
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=128 --train_path /high_perf_store3/l3_data/gent/data/ffcv/IN1K_train_smart.ffcv --val_path /high_perf_store3/l3_data/gent/data/ffcv/IN1K_val_smart.ffcv --ckpt_freq 5 --opt adamw --opt_betas 0.9 .999 --blr 5e-4 --reprob 0.25 --mixup 0.8 --cutmix 1.  --weight_decay 0.05 --layer_decay=0.65 --warmup_epochs=5 --epochs=100 --ThreeAugment --model vit_base_patch16_224 --gin build_model.drop_path_rate=0.1 --output_dir outputs/cls/MAE_base-IN1K_s4 --checkpoint_key model -w ../models/vitb/mae_pretrain_vit_base.pth 
#83.12 https://wandb.ai/erow/vitookit/runs/r464hnwc?nw=nwusererow


name=dinov3_vits-in1k
WANDB_NAME=${name} vitrun --nproc_per_node=8 vitookit/evaluation/eval_cls.py  --data_location $IMNET --data_set IN1K --ckpt_freq 5 --opt adamw --opt_betas 0.9 .999 --blr 5e-4 --smoothing=0.1 --warmup_epochs=5 --epochs=100  --model dino_vit_small --gin build_model.drop_path_rate=0.1 build_model.n_storage_tokens=4 build_model.layerscale_init=1e-4 -w ../outputs/dinov3/dinov3_vits16_pretrain_lvd1689m.pth --output_dir  ../experiments/dinov3_vits/baseline
```

## forgetting

```bash
export WANDB_TAGS="forget"

WANDB_NAME=dinov2_vitb-baseline sbatch --nodes=1 svitrun.sh vitookit/evaluation/eval_cls.py --batch_size=128 --data_location $IMNET  --ckpt_freq 5 --opt adamw --opt_betas 0.9 .999 --blr 1e-4  --smoothing=0.1 --warmup_epochs=5 --epochs=100  --gin build_model.pretrained=True build_model.dynamic_img_size=True --model vit_base_patch14_dinov2.lvd142m --output_dir ../experiments/ft/dinov2_vitb-baseline

WANDB_NAME=dinov2_vitb-baseline sbatch --nodes=1 svitrun.sh vitookit/evaluation/eval_cls.py --batch_size=128 --data_location $IMNET  --ckpt_freq 5 --opt adamw --opt_betas 0.9 .99 --blr 5e-5 --disable_weight_decay_on_bias_norm  --smoothing=0.1 --warmup_epochs=5 --epochs=100  --gin build_model.pretrained=True build_model.dynamic_img_size=True --model vit_base_patch14_dinov2.lvd142m --output_dir ../experiments/ft/dinov2_vitb-baseline

WANDB_NAME=dinov2_vitb-ld0.65 sbatch --nodes=1 svitrun.sh vitookit/evaluation/eval_cls.py --batch_size=128 --data_location $IMNET  --ckpt_freq 5 --opt adamw --opt_betas 0.9 .999 --blr 5e-4 --layer_decay=0.65  --smoothing=0.1 --warmup_epochs=5 --epochs=100  --gin build_model.pretrained=True build_model.dynamic_img_size=True --model vit_base_patch14_dinov2.lvd142m --output_dir ../experiments/ft/dinov2_vitb-ld0.65


WANDB_NAME=dinov2_vitb-ld0.75_dp0.1 sbatch --nodes=1 svitrun.sh vitookit/evaluation/eval_cls.py --batch_size=128 --data_location $IMNET  --ckpt_freq 5 --opt adamw --opt_betas 0.9 .999 --blr 5e-4  --layer_decay=0.75 --smoothing=0.1 --warmup_epochs=5 --epochs=100  --gin  build_model.dynamic_img_size=True build_model.drop_path_rate=0.1 build_model.pretrained=True --model vit_base_patch14_dinov2.lvd142m --output_dir ../experiments/ft/dinov2_vitb-dinov2_vitb-ld0.75_dp0.1


WANDB_NAME=dinov2_vitb-ld0.9_dp0.1 sbatch --nodes=1 svitrun.sh vitookit/evaluation/eval_cls.py --batch_size=128 --data_location $IMNET  --ckpt_freq 5 --opt adamw --opt_betas 0.9 .999 --blr 5e-4  --layer_decay=0.9 --smoothing=0.1 --warmup_epochs=5 --epochs=100  --gin   build_model.drop_path_rate=0.1 build_model.pretrained=True --model vit_base_patch14_dinov2.lvd142m --output_dir ../experiments/ft/dinov2_vitb-dinov2_vitb-ld0.9_dp0.1
```