# ViT tookit

We aim to provide a tookit for evaluating and analyzing Vision Transformers.
Install the package by

```bash
git clone https://github.com/erow/vitookit.git
pip install -e vitookit
# pip install git+https://github.com/erow/vitookit.git
```


# Run on HPC
See the available evaluations in [evaluation protocols](EVALUATION.md).

## commands

```bash
vitrun train_cls.py --data_location=../data/IMNET --gin build_model.model_name='"vit_tiny_patch16_224"' build_model.global_pool='"avg"'  -w wandb:dlib/EfficientSSL/ezuz0x4u --layer_decay=0.75 
```

## condor

```bash
condor_submit condor/eval_weka_cls.submit model_dir=outputs/dinosara/base ARCH=vit_base
```

## Slurm

```text

usage: submitit for evaluation [-h] [--module MODULE] [--ngpus NGPUS] [--nodes NODES] [-t TIMEOUT] [--mem MEM] [--partition PARTITION] [--comment COMMENT] [--job_dir JOB_DIR] [--fast_dir FAST_DIR]

options:
  -h, --help            show this help message and exit
  --module MODULE       Module to run
  --ngpus NGPUS         Number of gpus to request on each node
  --nodes NODES         Number of nodes to request
  -t TIMEOUT, --timeout TIMEOUT
                        Duration of the job
  --mem MEM             Memory to request
  --partition PARTITION
                        Partition where to submit
  --comment COMMENT     Comment to pass to scheduler
  --job_dir JOB_DIR
  --fast_dir FAST_DIR   The dictory of fast disk to load the datasets

```

We move files to **FAST_DIR**. For example, to finetune a pre-trained model on ImageNet, run:
```bash
submitit  --module vitookit.evaluation.eval_cls_ffcv   --train_path  ~/data/ffcv/IN1K_train_500_95.ffcv --val_path  ~/data/ffcv/IN1K_val_500_95.ffcv --fast_dir /raid/local_scratch/jxw30-hxc19/ --gin VisionTransformer.global_pool='"avg"' -w wandb:dlib/EfficientSSL/lsx2qmys 
```

# Evaluation

There are many protocols for evaluating the performance of a model. We provide a set of evaluation scripts for different tasks. Use `vitrun` to launch the evaluation.

**Fineune Protocol for Image Classification**

``vitrun eval_cls.py --data_location=$data_path -w <weights.pth> --gin key=value``

**Lnear prob Protocol for Image Classification**

``vitrun eval_linear.py --data_location=$data_path -w <weights.pth> --gin key=value``

More evaluation scripts can be found in [evaluation](evaluation/README.md).

## Flexible Configuration

We use gin-config to configure the model and the training process. You can easily change the configuration by passing the gin files by `--cfgs <file1> <file2> ...` or directly change the bindings by `--gin key=value`.

## Pretrained weights

The pretrained weights can be one of the following:

- a local file
- a url starting with `https://`
- an artifact path starting with `artifact:`
- a run path starting with `wandb:<entity>/<project>/<run>`, where `weights.pth` will be used as the weights file.

You can further specify the *key* and *prefix* to extract the weights from a checkpoint file. For example, `--pretrained_weights=ckpt.pth --checkpoint_key model --prefix module.` will extract the state dict from the key "model" in the checkpoint file and remove its prefix "module." in the keys.



## cluster

To see the self-attention map and the feature map of a given image, run  

`` python bin/viz_vit.py --arch vit_base --pretrained_weights <checkpoint.pth> --img imgs/sample1.JPEG``

## CAM Visualization
Gram CAM is an important tool to diagnose model predictions. We use [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) to visualize the parts focused by the model for classification. 

```
python bin/grad_cam.py --arch=vit_base  --method=scorecam --pretrained_weights=<> --img imgs/sample1.JPEG--output_img=<>
```

# Evaluation
## Condor

Run a group of experiments:

```
condor_submit condor/eval_stornext_cls.submit model_dir=../SiT/outputs/imagenet/sit-ViT_B head_type=0
```