from vitookit.models import mvit, dino_vit, dmes, crisp
 
try:
    import vitookit.models.custom
    import vitookit.models.cvit
except ImportError as e:
    print('warning: custom models not found', e)
    pass
