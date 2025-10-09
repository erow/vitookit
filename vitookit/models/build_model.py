import gin
import timm
# try:
import os
import importlib.util
import glob

from torch import nn

@gin.configurable()
def create_backbone(model_name, **kwargs):
    if isinstance(model_name,str):
        backbone = timm.create_model(model_name, num_classes=0, **kwargs)
    else:
        backbone = model_name(**kwargs)
    return backbone

@gin.configurable()
def build_model(model_name, **kwargs):
    try:
        # try build model with timm
        model = timm.create_model(model_name, **kwargs)        
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        model = model_name(**kwargs)
    return model
    

def build_head(num_layers, input_dim, mlp_dim, output_dim, hidden_bn=True,activation=nn.ReLU,
               last_norm='bn',):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        if l == num_layers-1:
            mlp.append(nn.Linear(dim1, dim2, bias=False))
        else:
            mlp.append(nn.Linear(dim1, dim2, bias=True))

        if l < num_layers - 1:
            if hidden_bn:
                mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(activation())
        else:
            if last_norm=='bn':
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
            elif last_norm=='ln':
                mlp.append(nn.LayerNorm(dim2))
            elif last_norm=='none':
                pass
            else:
                raise NotImplementedError(f"last_norm={last_norm} not implemented")

    return nn.Sequential(*mlp)
