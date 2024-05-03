import gin


def build_ffcv():
    order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    data_loader_train =  Loader(args.train_path, pipelines=ThreeAugmentPipeline(),batches_ahead=1,
                        batch_size=args.batch_size, num_workers=args.num_workers, 
                        order=order, distributed=args.distributed,seed=args.seed)
    

    data_loader_val =  Loader(args.val_path, pipelines=ValPipeline(),
                        batch_size=args.batch_size, num_workers=args.num_workers, batches_ahead=1,
                        distributed=args.distributed,seed=args.seed)


@gin.configurable()
class DataLoader():
    def __init__(self, data_set='ffcv') -> None:
        pass