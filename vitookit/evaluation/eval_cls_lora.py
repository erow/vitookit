"""
finetuning LoRA adapters for image classification tasks.
Run: vitrun eval_cls_lora.py --data_set CIFAR10 --model vit_tiny_patch16_224 --gin build_model.pretrained=True --cfgs cifar.gin

Parameters:
- r: rank of the LoRA adapters
- lora_alpha: scaling factor
- lora_dropout: dropout rate for LoRA adapters
- target_modules: regex pattern to match module names to apply LoRA. e.g. 'mlp' to match the FFN modules, 'attn' to match attention modules.
"""
from vitookit.evaluation import eval_cls  
from vitookit.models.lora import apply_lora

_train = eval_cls.train
def pre_train(args,model,data_loader_train, data_loader_val):
    lora_modules = apply_lora(model, args.lora_target_modules, args.lora_r, args.lora_alpha, args.lora_dropout)
    print("lora ", lora_modules)
    _train(args,model,data_loader_train, data_loader_val)

if __name__ == '__main__':
    parser = eval_cls.get_args_parser()
    parser.add_argument('--lora_target_modules', type=str, default='attn',
                        help='Target modules to apply LoRA, as a regex pattern string.')
    parser.add_argument('--lora_r', type=int, default=4,
                        help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                        help='LoRA alpha scaling factor.')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout rate.') 
    args = parser.parse_args()
    # hack to replace train function
    eval_cls.train = pre_train
    eval_cls.main(args)