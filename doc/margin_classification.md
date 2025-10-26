# Large Margin Classification

```latex
@INPROCEEDINGS{9207251,
  author={Wu, Jiantao and Wang, Lin},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={ArcGrad: Angular Gradient Margin Loss for Classification}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},
  keywords={Training;Task analysis;Neural networks;Face recognition;Feature extraction;Acceleration;Tuning;loss function;margin;angular;gradient;adacos;softmax;ArcFace},
  doi={10.1109/IJCNN48605.2020.9207251}}

```

Run:
```bash
# 96.35, https://wandb.ai/erow/margin/runs/15xo55eq/
WANDB_NAME=arcgrad_vitt_e200 vitrun eval_cls_margin.py  --ckpt_freq 2 --opt adamw --lr 0.001 --weight_decay 0.05   --batch_size=256 --ra=3 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --model vit_tiny_patch16_224 --gin "build_transform.scale=(0.8,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" build_model.patch_size=2 build_model.img_size=32  --margin_loss arcgrad

# 96.21，https://wandb.ai/erow/margin/runs/dtktrgov/overview
WANDB_NAME=arcgrad sbatch svitrun.sh eval_cls_margin.py --ckpt_freq 2 --opt muon --lr 0.002 --weight_decay 5e-4 --warmup_epochs=10 --epochs 100 --batch_size=128 --ra=3 --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 64 --model resnet18 --gin build_transform.scale=(0.8,1) "build_transform.mean=(0.4914, 0.4822, 0.4465)" "build_transform.std=(0.2470, 0.2435, 0.2616)" build_model.stem_type='deep' build_model.output_stride=8 MarginHead.s=10 --margin_loss arcgrad 

```
margin_loss options: arcface, softmax, arcgrad.

## ArcGrad
Angular Gradient Margin Loss (ArcGrad), which promotes intra-class compactness and inter-class separability by generating a "gradient margin". Key contributions of the paper include:

1. It suggests that the margin parameter found in methods like ArcFace is not necessary.
2. It establishes that the scaling parameter and the margin are inversely proportional, simplifying the tuning process.

Instead of creating a hard margin by changing the decision boundary, ArcGrad generates a soft margin by maximizing the angular gradient of features in a specific region, causing them to move faster during training.


**Angular Gradient Margin Loss (ArcGrad)**: The proposed loss function, which uses the angle $\theta$ directly, rather than its cosine, and removes the explicit margin parameter.
    $$\mathcal{L}^{ArcGrad}=-\frac{1}{N}\sum_{i=1}^{N}log\frac{e^{-s\cdot\theta_{y_{i}}}}{\sum_{j=1}^{C}e^{-s\cdot\theta_{j}}}$$

**Gradient of Approximated Loss**: The function of the ArcGrad loss's gradient with respect to the average angular gap $\tilde{m}$. The objective is to select a scaling parameter $s$ that maximizes this gradient.
    $$\frac{\partial L^{ArcGrad^{\prime\prime}}}{\partial\tilde{m}}=\frac{(C-1)e^{\tilde{m}\cdot s}s}{(C-1)e^{\tilde{m}\cdot s}+1}$$

**Scaling Parameter and Margin Relationship**: By solving for the maximum of the gradient function, the paper establishes an inverse relationship between the scaling parameter $s$ and the average angular gap $\tilde{m}$.
    $$s = \frac{ProductLog(C-1)+1}{\tilde{m}}$$
    where `ProductLog` is the Lambert W function.

### **Evaluation Metrics**

To measure intra-class compactness and inter-class separability, the paper uses three metrics based on the angles between weight vectors ($W$) and the mean feature vectors for each class ($Center$).

* **WC-Intra**: The mean angle between the feature center and the weight vector of the same class.
    $$\mathrm{WC-Intra}=\frac{1}{C}\sum_{j=1}^{C}\langle Center_{j},W_{j}\rangle$$

* **W-Inter**: The mean angle between the weight vectors of different classes.
    $$\mathrm{W-Inter}=\frac{1}{C^{2}-C}\sum_{j=1}^{C}\sum_{i=1,i\ne j}^{C}\langle W_{i},W_{j}\rangle$$

* **C-Inter**: The mean of angles between different classes' feature centers.
    $$\mathrm{C-Inter}=\frac{1}{C^{2}-C}\sum_{j=1}^{C}\sum_{i=1,i\ne j}^{C}\langle Center_{j},W_{j}\rangle$$