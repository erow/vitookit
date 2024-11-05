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
    parser.add_argument("--output", default="resnet50.pth", type=str, help="the new name of the weights after extraction.")
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
            i=0
            print(f"ckpt keys-{i}:", state_dict.keys())
            if args.ckpt_key:
                for k in args.ckpt_key.split("/"):
                    state_dict=state_dict[k]
                    i += 1
                    print(f"ckpt keys-{i}:", state_dict.keys())
            if args.prefix:
                pattern = re.compile(args.prefix)
                new_state_dict = {}
                for k,v in state_dict.items():
                    m = replace_key(pattern, k)
                    if m:
                        new_state_dict[m]=v
            file_new = args.output
            print("converting", file, "to", file_new)
            print(new_state_dict.keys())
            torch.save(new_state_dict,file_new)