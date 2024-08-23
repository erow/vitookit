
from ffcv import Loader
import gin
from .ffcv_transform import ThreeAugmentPipeline

DEFAULT_SCHEMES ={
    0: [
        dict(res=160,lower_scale=0.08, upper_scale=1,color_jitter=False),
        dict(res=176,lower_scale=0.08, upper_scale=1,color_jitter=False),
        dict(res=192,lower_scale=0.08, upper_scale=1,color_jitter=True),
    ],
    1: [
        dict(res=160,lower_scale=0.08, upper_scale=1,color_jitter=False),
        dict(res=192,lower_scale=0.08, upper_scale=1,color_jitter=False),
        dict(res=224,lower_scale=0.08, upper_scale=1,color_jitter=True),
    ],
    2:[
        dict(res=160,lower_scale=0.96, upper_scale=1,color_jitter=False),
        dict(res=192,lower_scale=0.38, upper_scale=1,color_jitter=False),
        dict(res=224,lower_scale=0.08, upper_scale=1,color_jitter=True),
    ],
    3:[
        dict(res=160,lower_scale=0.08, upper_scale=0.4624,color_jitter=False),
        dict(res=192,lower_scale=0.08, upper_scale=0.7056,color_jitter=False),
        dict(res=224,lower_scale=0.08, upper_scale=1,color_jitter=True),
    ],
    4:[
        dict(res=160,lower_scale=0.20, upper_scale=0.634,color_jitter=False),
        dict(res=192,lower_scale=0.137, upper_scale=0.81,color_jitter=False),
        dict(res=224,lower_scale=0.08, upper_scale=1,color_jitter=True),
    ],
    5: [
        dict(res=160,lower_scale=0.08, upper_scale=1,color_jitter=True),
        dict(res=192,lower_scale=0.08, upper_scale=1,color_jitter=True),
        dict(res=224,lower_scale=0.08, upper_scale=1,color_jitter=True),
    ],
    
}

@gin.configurable
class DynamicResolution:
    def __init__(self, start_ramp=65, end_ramp=80,  
                    scheme=0):
        if isinstance(scheme, int):
            scheme = DEFAULT_SCHEMES[scheme]
        self.scheme = scheme
        self.start_ramp = start_ramp
        self.end_ramp = end_ramp
        
    
    def get_config(self, epoch):
        if epoch <= self.start_ramp:
            return self.scheme[0]
        elif epoch>=self.end_ramp:
            return self.scheme[-1]
        else:
            i = (epoch-self.start_ramp) * (len(self.scheme)-1) // (self.end_ramp-self.start_ramp)
            return self.scheme[i]
    
    def __call__(self, loader: Loader, epoch,is_ffcv=False):
        config = self.get_config(epoch)
        print(", ".join([f"{k}={v}" for k,v in config.items()]))
        
        img_size = config['res']
        lower_scale = config['lower_scale']
        upper_scale = config['upper_scale']
        color_jitter = config['color_jitter']

        if is_ffcv:
            pipelines = ThreeAugmentPipeline(img_size,scale=(lower_scale,upper_scale),color_jitter=color_jitter)
            loader.compile_pipeline(pipelines)           
        else:
            # todo: change dres
            pipelines = loader.dataset.transforms
            
            decoder = loader.dataset.transforms.transform.transforms[0]
            decoder.size=(img_size,img_size)
            decoder.scale = (lower_scale,upper_scale)
