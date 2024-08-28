# Finetune

## ResNet


Train from scratch:
```bash
name=Random-IN1K # https://arxiv.org/pdf/2110.00476#page=5.70
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $train_path --val_path $val_path  --ckpt_freq 5 --opt LAMB --lr 1e-3 --weight_decay 2e-2 --smoothing=0 --gin ThreeAugmentPipeline.img_size=160 build_model.model_name="'resnet50'" --output_dir outputs/cls/${name} 

```

Finetuning:
```bash
name=Random-IN1K # https://arxiv.org/pdf/2110.00476#page=5.70
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $train_path --val_path $val_path  --ckpt_freq 5 --opt LAMB --lr 1e-3 --weight_decay 2e-2 --smoothing=0 --layer_decay 0.75 --gin ThreeAugmentPipeline.img_size=160 build_model.model_name="'resnet50'" --output_dir outputs/cls/${name} 

WANDB_NAME=$name  vitrun --nproc_per_node 8 timm_train.py ~/data/ImageNet -b 64 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce   --pretrained-path ../SSL_dev/outputs/pretrain/HydraV2_e1000_IN1K/resnet50.pth --pretrained --layer-decay 0.75 --log-wandb  --output outputs/cls/${name}



```


## ViT
```bash
name=vitb_Random-IN1K 
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $train_path --val_path $val_path  --ckpt_freq 5 --opt AdamW --opt_betas 0.9 .95 --lr 1e-3 --weight_decay 1e-2 --smoothing=0.1 --gin ThreeAugmentPipeline.img_size=160 build_model.model_name="'vit_base_patch16_224'" build_model.drop_path=0.1 --output_dir outputs/cls/${name} 

```

