import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import gin

try:
    from scipy.special import lambertw
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    lambertw = None
    
@gin.configurable()
class ArcGrad(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            num_classes: number of classes
            s: scale factor
            margin: soft margin
        """
    def __init__(self, num_classes, s):
        super(ArcGrad, self).__init__()
        self.num_classes = num_classes
        if SCIPY_AVAILABLE:
            productlog = lambertw((num_classes-1)).real
            margin = (productlog+1)/s
        else:
            # Fallback approximation for lambertw
            x = (num_classes-1)
            margin = (np.log(x) + 1)/s
        self.margin = margin
        self.s = s
        
    def __repr__(self):
        return f'ArcGrad(s={self.s:.3f}, margin={self.margin})'

    def forward(self, cosine,labels):
        theta = 3.1415927 - cosine.acos()
        output = self.s * theta
        return output

class Softmax(nn.Module):
    r"""Implement of softmax loss:
        Args:
            num_classes: number of classes
            s: scale factor
        """
    def __init__(self, num_classes, s):
        super(Softmax, self).__init__()
        self.num_classes = num_classes
        self.s = s
    
    def __repr__(self):
        return f'Softmax(s={self.s:.3f})'
    
    def forward(self, cosine: torch.Tensor, labels: torch.Tensor):
        output = self.s * cosine
        return output

@gin.configurable()
class ArcFace(nn.Module):
    r"""Implement of arcface loss:
        Args:
            num_classes: number of classes
            s: scale factor
        """
    def __init__(self, num_classes, s, margin=0.5):
        self.num_classes = num_classes
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
    
    def __repr__(self):
        return f'ArcFace(s={self.s:.3f}, margin={self.margin})'

    def forward(self, cosine: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = cosine[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            cosine.arccos_()
            final_target_logit = target_logit + self.margin
            cosine[index, labels[index].view(-1)] = final_target_logit
            cosine.cos_()
        cosine = cosine * self.s   
        return cosine

@gin.configurable()
class MarginHead(nn.Module):
    r"""Implement of classification head with margin loss:
        Args:
            in_features: size of each input sample
            num_classes: number of classes
        """
    def __init__(self, in_features, num_classes, 
                 s = 10,
                 margin_loss='arcgrad',
                 embed_dim=512):
        super(MarginHead, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, embed_dim*4, bias=False),
            nn.ReLU(), nn.BatchNorm1d(embed_dim*4),
            nn.Linear(embed_dim*4, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False),
        )
        self.weight = nn.utils.weight_norm(
            nn.Linear(embed_dim, num_classes, bias=False)
        )
        self.weight.weight_g.requires_grad = False # freeze the magnitude of the weight        
        self.weight.weight_g.fill_(1)
        
        if margin_loss == 'arcgrad':
            self.margin = ArcGrad(num_classes,s)
        elif margin_loss == 'arcface':
            self.margin = ArcFace(num_classes,s)
        elif margin_loss == 'softmax':
            self.margin = Softmax(num_classes,s)
        else:
            raise ValueError(f'Invalid margin loss: {margin_loss}')
    
    def get_weight(self):
        return self.weight.weight_v.detach()

    def forward(self, input, labels=None,return_z=False):
        z = self.fc(input)
        z = F.normalize(z, dim=1)
        cosine = self.weight(z)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        if self.training:
            logits = self.margin(cosine,labels)
        else:
            logits = cosine
        if return_z:
            return logits, z
        else:
            return logits
    