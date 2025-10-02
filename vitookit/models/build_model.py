import gin
import timm
# try:
import os
import importlib.util
import glob

from torch import nn
# models = os.environ.get('CUSTOM_MODELS', 'custom_models')
# custom_model_files = glob.glob(f"{models}/*.py")
# for file in custom_model_files:
#     module_name = os.path.splitext(os.path.basename(file))[0]
#     spec = importlib.util.spec_from_file_location(module_name, file)
#     module = importlib.util.module_from_spec(spec)
#     print("import custom model:", module_name, spec)        
#     spec.loader.load_module(module_name)

# except ImportError:
#     print("No custom models found, using timm models.")


@gin.configurable()
def create_backbone(model_name, **kwargs):
    if isinstance(model_name,str):
        backbone = timm.create_model(model_name, **kwargs)
    else:
        backbone = model_name(**kwargs)
    return backbone

@gin.configurable()
def build_model(model_name, **kwargs):
    try:
        # try build model with timm
        model = create_backbone(model_name, **kwargs)        
    except:
        model = model_name(**kwargs)
    return model
    
    