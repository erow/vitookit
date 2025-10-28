from vitookit.models import mvit, dino_vit, dmes, crisp
from vitookit.models import vit_extra  
try:
    import vitookit.models.custom
except ImportError as e:
    print('warning: custom models not found', e)
    pass
