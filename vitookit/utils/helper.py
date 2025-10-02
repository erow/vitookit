import argparse, os, sys
import json
from pathlib import Path
import numpy as np
import einops
import torch
import gin


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def post_args(args):
    import gin
    import yaml
    if not gin.config_is_locked() and hasattr(args, "cfgs") and hasattr(args, "gin"):
        gin.parse_config_files_and_bindings(args.cfgs,args.gin)

    if hasattr(args,"output_dir") and args.output_dir:
        output_dir=Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(os.path.join(args.output_dir,'config.yml'), 'w') as f:
            yaml.dump(vars(args), f)
            
        open(output_dir/"config.gin",'w').write(gin.config_str())

def patchify(imgs, p =16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    x = einops.rearrange(imgs,'n c(h p1)(w p2)->n(h w)(p1 p2 c)',p1=p,p2=p)
    return x

def unpatchify(x, p=16):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = int(np.sqrt(x.shape[1]))
    x = einops.rearrange(x,'n(h w)(p1 p2 c) -> n c(h p1)(w p2)',p1=p,p2=p,h=h)
    return x

def log_metrics(prefix, metrics: dict, args):
    if args.output_dir:
        os.makedirs(args.output_dir,exist_ok=True)
        with open(os.path.join(args.output_dir,f"{prefix}.json"),'a') as f:
            f.write('\n'+json.dumps(dict(metrics)))
    if 'wandb:' in args.pretrained_weights:
        import wandb
        api = wandb.Api()
        run = api.run(args.pretrained_weights.split(":")[1])        
        for k,v in metrics.items():
            run.summary[f"{prefix}/{k}"]=v

        run.update()

def log_file(filename,state,args):
    if args.output_dir:
        os.makedirs(args.output_dir,exist_ok=True)
        if '.npy' in filename:
            np.save(os.path.join(args.output_dir,filename),state,allow_pickle=True)
        elif ".pth" in filename:
            torch.save(state,os.path.join(args.output_dir,filename))
        else:
            json.dump(state,open(os.path.join(args.output_dir,filename),'w'))
        
    if 'wandb:' in args.pretrained_weights:
        import wandb
        if wandb.run is None:
            run=wandb.init(job_type='upload')
        else:
            run = wandb.run
        
        artifact = run.use_artifact(args.pretrained_weights[6:]+":latest", type='model')
        draft_artifact = artifact.new_draft()
        draft_artifact.alias = args.pretrained_weights.split(":")[1]
        draft_artifact.add_file(os.path.join(args.output_dir,filename))
        # draft_artifact.name = artifact.name
        run.log_artifact(draft_artifact)
        run.link_artifact(draft_artifact, args.pretrained_weights.split(":")[1])


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        print("the file doesn't exist")
        return
    print("Found checkpoint at {}".format(ckp_path))
    if ckp_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            ckp_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(ckp_path, map_location='cpu',weights_only=False)
        
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            if key == "model_ema":
                value.ema.load_state_dict(checkpoint[key])
            else:
                value.load_state_dict(checkpoint[key])
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def interpolate_pos_embed(model, model_checkpoint):
    if 'pos_embed' in model_checkpoint:
        pos_embed_checkpoint = model_checkpoint['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            model_checkpoint['pos_embed'] = new_pos_embed

def wandb_download(path):
    path = path.replace("wandb:","")
    model_hub = os.getenv("MODEL_HUB", "./MODEL_HUB")
    model_path = os.path.join(model_hub, path)
    os.makedirs(model_path, exist_ok=True)
    basename = os.path.basename(model_path)
    if not os.path.exists(model_path):
        import wandb
        api = wandb.Api()
        run = api.run(path)
        file_path = run.file(basename).download(root=model_path, replace=True).name # download weights to current dir         
        print("Download checkpoint path: %s" % (file_path))
    else:
        print("Use cached path: %s" % model_path)
    return model_path

def load_pretrained_weights(model, pretrained_weights, 
                            checkpoint_key=None, prefix=None,interpolate=False):
    """load vit weights"""
    if pretrained_weights == '':
        return
    elif pretrained_weights.startswith('https'):
        state_dict = torch.hub.load_state_dict_from_url(
            pretrained_weights, map_location='cpu', check_hash=True)
    elif pretrained_weights.startswith('artifact:'):
        path = pretrained_weights.replace("artifact:","")
        import wandb
        import tempfile
        api = wandb.Api()        
        artifact = api.artifact(path, type='model')
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = artifact.download(os.getenv("output_dir",tmp_dir))
            file_path = os.path.join(artifact_dir, "weights.pth")
            print("Real checkpoint path: %s" % (file_path))
            state_dict = torch.load(file_path, map_location='cpu')
    elif pretrained_weights.startswith('wandb:'):
        path = pretrained_weights.replace("wandb:","")
        model_hub = os.getenv("MODEL_HUB", "./MODEL_HUB")
        model_path = os.path.join(model_hub, path)
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(os.path.join(model_path, "weights.pth")):
            import wandb
            api = wandb.Api()
            run = api.run(path)
            file_path = run.file("weights.pth").download(root=model_path, replace=True).name # download weights to current dir         
            print("Download checkpoint path: %s" % (file_path))
        else:
            file_path = os.path.join(model_path, "weights.pth")
            print("Use cached checkpoint path: %s" % (file_path))    
        state_dict = torch.load(file_path, map_location='cpu')
    elif os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location='cpu',weights_only=False)
    else:
        raise ValueError(f'load pretrained weights from {pretrained_weights} failed!')    
    
    epoch = state_dict['epoch'] if 'epoch' in state_dict else -1
    print("Load pre-trained checkpoint from: %s[%s] at %d epoch" % (pretrained_weights, checkpoint_key, epoch))
    
    if checkpoint_key:
        state_dict = state_dict[checkpoint_key]
    # remove prefix
    if prefix:
        import re
        pattern = re.compile(prefix)
        state_dict = {pattern.match(k).group(1): v for k, v in state_dict.items()  if pattern.match(k)}
    
    if interpolate and 'pos_embed' in state_dict:
        # interpolate position embedding
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        print('interpolate pos_embed from {} to {}'.format(pos_embed_checkpoint.shape, model.pos_embed.shape))
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] ) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # print('debug:', pos_embed_checkpoint.shape,orig_size,new_size,num_extra_tokens)
        
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))        