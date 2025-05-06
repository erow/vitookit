import gin
import timm
# try:
import os
import importlib.util
import glob
models = os.environ.get('CUSTOM_MODELS', 'custom_models')
custom_model_files = glob.glob(f"{models}/*.py")
for file in custom_model_files:
    module_name = os.path.splitext(os.path.basename(file))[0]
    spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(spec)
    print("import custom model:", module_name, spec)        
    spec.loader.load_module(module_name)

# except ImportError:
#     print("No custom models found, using timm models.")

@gin.configurable()
def build_model(model_name='vit_base_patch16_224', **kwargs):
    if isinstance(model_name,str):
        model = timm.create_model(model_name, **kwargs)
    else:
        model = model_name(**kwargs)
    return model
