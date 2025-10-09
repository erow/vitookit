# Simple Representation Learning from Arbitrary Pairs

Representation learning traditionally follows a simple principle: pull semantically similar samples together and push dissimilar ones apart. 
This principle underlies most existing approaches, including supervised classification, self-supervised learning, and contrastive methods, and it has been central to their success. Yet it overlooks an important source of information: Even when classes appear unrelated, their samples often share latent visual attributes such as shapes, textures, or structural patterns. For example, cats, dogs and cattle have fur and four limbs etc. These overlooked commonalities raise a fundamental question: **can models learn from arbitrary pairs without explicit guidance?**


We show that the answer is yes. The primary challenge lies in learning from dissimilar samples while preserving the notion of semantic distance. We resolve this by proving that for any pair of classes, there exists a subspace where their shared features are discriminative to other classes. 
To uncover these subspaces we propose SimLAP, a Simple framework to Learn from Arbitrary Pair. SimLAP uses a lightweight feature filter to adaptively activate shared attributes for any given pair.
Through extensive experiments we show that models trained via SimLAP can indeed learn effectively from arbitrary pairs. 
Remarkably, models learned from arbitrary pairs are more transferable than those learned from traditional representation learning methods and exhibit greater resistance to representation collapse. 
Our findings suggest that arbitrary pairs, often dismissed as irrelevant, are in fact a rich, complementary and untapped source of supervision. By learning from them we move beyond rigid notions of similarity. Hopefully, SimLAP will open an additional pathway toward more general and robust representation learning.


```tex
@article{wu2024rethinking,
  title={Rethinking positive pairs in contrastive learning},
  author={Wu, Jiantao and Atito, Sara and Feng, Zhenhua and Mo, Shentong and Kitler, Josef and Awais, Muhammad},
  journal={arXiv preprint arXiv:2410.18200},
  year={2024}
}
```

## Run

```bash
# pretrain:  90.86
vitrun simlap.py --ckpt_freq 2 --opt adam --lr 0.001 --weight_decay 5e-4  --warmup_epochs=2 --epochs 50  --batch_size=256  --ra=3  --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --model resnet18 --gin "build_transform.scale=(0.7,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" create_backbone.stem_type="'deep'" create_backbone.output_stride=8  SimLAP.embed_dim=512 --output_dir outputs/simlap/cifar10_resnet18

# ft
vitrun eval_cls.py  --ckpt_freq 2 --opt sgd --lr 0.1  --layer_decay=0.65  --weight_decay 5e-4 --warmup_epochs=10 --epochs 300 --batch_size=256  --ra=3  --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --model resnet18 --gin "build_transform.scale=(0.7,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" build_model.stem_type="'deep'" build_model.output_stride=8 --checkpoint_key model --prefix='backbone.(.*)' -w outputs/simlap/r18_e50/checkpoint.pth --output_dir outputs/simlap/cifar10_resnet18/ft

# fuse
vitrun simlap_fuse.py  --ckpt_freq 2 --opt sgd --lr 0.1  --weight_decay 5e-4 --warmup_epochs=10 --epochs 300 --batch_size=256  --ra=3  --smoothing=0.1 --reprob 0.1 --data_set CIFAR10 --data_location ../data --input_size 32 --model resnet18 --gin "build_transform.scale=(0.7,1)" build_transform.mean="(0.4914, 0.4822, 0.4465)" build_transform.std="(0.2470, 0.2435, 0.2616)" create_backbone.stem_type="'deep'" create_backbone.output_stride=8 SimLAPFuse.embed_dim=512 --output_dir outputs/simlap/cifar10_resnet18_fuse
```

