# Vitookit - Vision Model Evaluation Toolkit

A comprehensive toolkit for evaluating and analyzing vision models, with a focus on Vision Transformers (ViT). Vitookit provides flexible evaluation protocols, distributed training support, and seamless integration with HPC clusters and Weights & Biases.

## Features

- 🎯 **Multiple Evaluation Protocols**: Full finetuning, linear probing, k-NN, and more
- ⚡ **Fast Data Loading**: Support for both PyTorch DataLoader and FFCV for high-performance I/O
- 🔧 **Flexible Configuration**: Gin-config based configuration system
- 🚀 **Distributed Training**: Multi-GPU and multi-node training with PyTorch DDP
- 🏭 **HPC Integration**: Slurm and Condor cluster support with automatic job management
- 📊 **WandB Integration**: Seamless experiment tracking and model versioning
- 🗂️ **Various Datasets**: ImageNet variants, CIFAR, transfer learning datasets, and more

## Installation

```bash
pip install git+https://github.com/erow/vitookit.git
```

## Quick Start

### Local Training

`vitrun` is a wrapper around `torchrun` that automatically locates evaluation scripts in `vitookit/evaluation`.

**Single GPU:**
```bash
vitrun eval_cls.py \
  --data_location=/data/IMNET \
  --model vit_tiny_patch16_224 \
  --gin build_model.global_pool='"avg"'
```

**Multi-GPU (distributed):**
```bash
vitrun --nproc_per_node=8 eval_cls.py \
  --data_location=/data/IMNET \
  --model vit_base_patch16_224 \
  --batch_size=128
```

### HPC Cluster Deployment

#### Slurm

Submit a job to a Slurm cluster:
```bash
sbatch hpc/svitrun.sh eval_cls.py \
  --data_location=/data/IMNET \
  --model vit_tiny_patch16_224 \
  --gin build_model.global_pool='"avg"'
```

#### Submitit (Recommended for HPC)

Submitit provides automatic checkpointing, job requeuing, and better cluster integration:

```bash
submitit \
  --module vitookit.evaluation.eval_cls \
  --ngpus=8 \
  --nodes=1 \
  --partition=gpu \
  --data_location=/data/IMNET \
  --model vit_base_patch16_224
```

**Available Options:**
- `--module`: Python module to run (e.g., `vitookit.evaluation.eval_cls`)
- `--ngpus`: Number of GPUs per node
- `--nodes`: Number of nodes to request
- `-t, --timeout`: Job duration in minutes
- `--mem`: Memory to request (GB)
- `--partition`: Cluster partition name
- `--job_dir`: Directory for job outputs
- `--fast_dir`: Fast disk directory for dataset caching

**FFCV Fast Loading Example:**
```bash
submitit \
  --module vitookit.evaluation.eval_cls_ffcv \
  --ngpus=8 \
  --train_path ~/data/ffcv/IN1K_train_500_95.ffcv \
  --val_path ~/data/ffcv/IN1K_val_500_95.ffcv \
  --fast_dir /scratch/local/ \
  -w wandb:dlib/EfficientSSL/lsx2qmys
```

#### Condor
```bash
condor_submit condor/eval.submit
```

## Evaluation Protocols

Vitookit provides multiple evaluation protocols for comprehensive model assessment. Use `vitrun` to launch any evaluation script.

### Full Finetuning

Train all model parameters end-to-end:
```bash
vitrun eval_cls.py \
  --data_location=$DATA_PATH \
  -w <weights.pth> \
  --gin build_model.drop_path_rate=0.1
```

See **[doc/finetune.md](doc/finetune.md)** for detailed training recipes and hyperparameter settings.

### Linear Probing

Freeze the backbone and train only the classification head:
```bash
vitrun eval_linear.py \
  --data_location=$DATA_PATH \
  -w <weights.pth> \
  --blr=0.1
```

### k-Nearest Neighbors

Parameter-free evaluation using feature similarity:
```bash
vitrun eval_knn.py \
  --data_location=$DATA_PATH \
  -w <weights.pth>
```

### Learning Rate Finder

Automatically determine optimal learning rates:
```bash
vitrun lr_finder.py \
  --data_location=$DATA_PATH \
  --model vit_base_patch16_224
```

## Configuration System

Vitookit uses [gin-config](https://github.com/google/gin-config) for flexible model and dataset configuration.

### Using Config Files
```bash
vitrun eval_cls.py --cfgs config.gin another_config.gin
```


### Command-line Overrides
```bash
vitrun eval_cls.py \
  --gin build_model.drop_path_rate=0.1 \
  --gin build_model.global_pool='"avg"'
```

**Note**: String values in gin require nested quotes: `'"value"'`

### Commonly Configured Functions
- `build_model()` - Model architecture and parameters
- `build_dataset()` - Dataset selection and preprocessing  
- `build_transform()` - Data augmentation strategies

## Pretrained Weights

Vitookit supports multiple sources for loading pretrained weights via `-w` or `--pretrained_weights`:

| Source Type | Format | Example |
|------------|--------|---------|
| **Local file** | File path | `-w /path/to/checkpoint.pth` |
| **HTTPS URL** | URL | `-w https://example.com/model.pth` |
| **WandB Run** | `wandb:<entity>/<project>/<run_id>` | `-w wandb:dlib/EfficientSSL/lsx2qmys` |
| **WandB Artifact** | `artifact:<entity>/<project>/<name>` | `-w artifact:dlib/models/vit-base` |

### Advanced Weight Loading

Extract weights from nested checkpoint structures:
```bash
vitrun eval_cls.py \
  -w checkpoint.pth \
  --checkpoint_key model \
  --prefix "^module\.(.*)"
```

**Options:**
- `--checkpoint_key`: Extract state dict from a specific key (e.g., `model`, `teacher`, `student`)
- `--prefix`: Regex pattern to remove from state dict keys (e.g., `module.` from DDP models)

## Supported Datasets

Vitookit supports 20+ datasets out of the box:

**ImageNet Variants:**
- `IN1K` - ImageNet-1K (ILSVRC2012)
- `IN100` - ImageNet-100 subset
- `IN1Kv2` - ImageNet-V2

**Transfer Learning Datasets:**
- `CIFAR10`, `CIFAR100` - CIFAR datasets
- `Flowers` - Oxford 102 Flowers
- `Cars` - Stanford Cars (196 classes)
- `Aircraft` - FGVC Aircraft
- `Pets` - Oxford-IIIT Pets (37 classes)
- `CUB200` - Caltech-UCSD Birds
- `Food` - Food-101
- `DTD` - Describable Textures Dataset
- `SUN397` - Scene Understanding
- `STL` - STL-10
- `INAT` - iNaturalist

**Custom:**
- `Folder` - Generic ImageFolder structure

## Key Training Parameters

### Learning Rate
- `--blr`: Base learning rate (scales with batch size: `lr = blr * batch_size / 256`)
- `--lr`: Absolute learning rate (overrides automatic scaling)
- `--layer_decay`: Layer-wise LR decay (0.5-0.75 recommended for ViT finetuning)

### Augmentation
- `--ThreeAugment`: Use 3-Augment instead of RandAugment
- `--mixup`: Mixup alpha (default: 0.8)
- `--cutmix`: Cutmix alpha (default: 1.0)
- `--ra N`: Repeated augmentation (N > 1 for batch augmentation)

### Common Training Recipes

**ViT from scratch:**
```bash
--opt adamw --blr 5e-4 --weight_decay 5e-2 --epochs 300
```

**ViT finetuning:**
```bash
--opt adamw --blr 5e-4 --layer_decay 0.65 --epochs 100
```

**ResNet50:**
```bash
--opt adamw --blr 5e-4 --weight_decay 2e-5
```

See [doc/finetune.md](doc/finetune.md) for more detailed recipes.

## Output & Logging

### Directory Structure
By default, outputs are saved to `--output_dir` (defaults to `outputs/<protocol>-<dataset>`):
- `checkpoint.pth` - Latest checkpoint (saved every `--ckpt_freq` epochs)
- `checkpoint_best.pth` - Best validation accuracy checkpoint
- `config.gin` - Gin configuration snapshot
- `config.yml` - Arguments configuration
- `log.txt` - JSON-formatted training logs

### Weights & Biases Integration

WandB is automatically initialized when `--output_dir` is set:
- Logs metrics, hyperparameters, and system information
- Supports automatic resuming from checkpoints
- Set custom run name: `WANDB_NAME=my_experiment vitrun ...`

## Documentation

- **[doc/finetune.md](doc/finetune.md)** - Detailed training recipes for different architectures
- **[doc/mvit.md](doc/mvit.md)** - Multiscale Vision Transformer specifics
- **[doc/simlap.md](doc/simlap.md)** - SimLap: Simple Representation Learning with Arbitrary Pairs
- **[doc/margin_classification.md](doc/margin_classification.md)** - Margin-based classification methods

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## Citation

If you use Vitookit in your research, please cite:
```bibtex
@software{vitookit,
  author = {Gent},
  title = {Vitookit: Vision Model Evaluation Toolkit},
  year = {2024},
  url = {https://github.com/erow/vitookit}
}
```

## License

This project is licensed under the MIT License.
