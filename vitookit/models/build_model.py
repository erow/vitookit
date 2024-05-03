import gin
import timm

@gin.configurable()
def build_model(model_name='vit_base_patch16_224', **kwargs):
    model = timm.create_model(model_name, **kwargs)
    return model
