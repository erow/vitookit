from vitookit.models import mvit, dino_vit, dmes
try:
    import vitookit.models.custom
except ImportError as e:
    print('warning: custom models not found', e)
    pass