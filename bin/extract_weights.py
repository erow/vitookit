#!/usr/bin/env python
import torch
import argparse
import glob
import re
import os

def get_parser():
    parser = argparse.ArgumentParser("Extract weights from checkpoints.")
    parser.add_argument("ckpt", type=str,help="the path of checkpoints, matching the path, e.g., '*/checkpoint.pth.'") 
    parser.add_argument("--prefix", type=str, default=None, help="the regex expression of the prefix in the weights.")
    parser.add_argument("--val", type=str, default=None, help="Validate the given string.")
    parser.add_argument("--ckpt_key", type=str, default=None, help="The path of the weights in the checkpoint.")
    parser.add_argument("--name", default="resnet50.pth", type=str, help="the new name of the weights after extraction.")
    return parser

def replace_key(pattern,s):
    groups = pattern.match(s)
    if groups:
        return groups.group(1)
    else:
        return None

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.val:
        pattern = re.compile(args.prefix)
        print(f"Test {args.val} -> {replace_key(pattern, args.val)}")
    else:
        for file in glob.glob(args.ckpt):
            state_dict = torch.load(file,"cpu")
            if args.ckpt_key:
                for k in args.ckpt_key.split("/"):
                    state_dict=state_dict[k]
            if args.prefix:
                pattern = re.compile(args.prefix)
                state_dict = {replace_key(pattern, k):v for k,v in state_dict.items()}
                del state_dict[None]
            file_new = file.replace(os.path.basename(file),args.name)
            print("converting", file, "to", file_new)
            print(state_dict.keys())
            torch.save(state_dict,file_new)