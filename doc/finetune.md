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

```

## ViT
```bash
name=vitb_Random-IN1K 
WANDB_NAME=${name} vitrun --nproc_per_node=8 eval_cls_ffcv.py --batch_size=256  --train_path $train_path --val_path $val_path  --ckpt_freq 5 --opt AdamW --opt_betas 0.9 .95 --lr 1e-3 --weight_decay 1e-2 --smoothing=0.1 --gin ThreeAugmentPipeline.img_size=160 build_model.model_name="'vit_base_patch16_224'" build_model.drop_path=0.1 --output_dir outputs/cls/${name} 

```