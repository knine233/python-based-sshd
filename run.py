import torch
import argparse
import os
from loguru import logger
import ssdh
from data.data_loader import load_data
from model_loader import load_model

multi_labels_dataset = [
    'nus-wide-tc-10',
    'nus-wide-tc-21',
    'flickr25k',
    'coco'
]

num_features = {
    'alexnet': 4096,
    'vgg16': 4096,
}


def run():
    # Load configuration
    args = load_config()
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")
    logger.info(args)

    # Load dataset
    query_dataloader, train_dataloder, retrieval_dataloader = load_data(args.dataset,
                                                                        args.root,
                                                                        args.num_query,
                                                                        args.num_train,
                                                                        args.batch_size,
                                                                        args.num_workers,
                                                                        )

    multi_labels = args.dataset in multi_labels_dataset
    if args.mode == "train":
        ssdh.train(
            train_dataloder,
            query_dataloader,
            retrieval_dataloader,
            multi_labels,
            args,
            num_features[args.arch],
        )
    elif args.mode == "resume":
        ssdh.train(
            train_dataloder,
            query_dataloader,
            retrieval_dataloader,
            multi_labels,
            args.code_length,
            num_features[args.arch],
            args.alpha,
            args.beta,
            args.max_iter,
            args.arch,
            args.lr,
            args.device,
            args.verbose,
            args.evaluate_interval,
            args.snapshot_interval,
            args.topk,
            args.checkpoint,
        )
    elif args.mode  == "evaluate":
        model = load_model(args.arch, args.code_length)
        model.load_snapshot(args.checkpoint)
        model.to(args.device)
        model.eval()
        mAP = ssdh.evaluate(
            model,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.topk,
            multi_labels,
            )
        logger.info('[Inference map:{:.4f}]'.format(mAP))
    else:
        raise ValueError('Error configuration, please check your config, using "train", "resume" or "evaluate".')


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='SSDH_PyTorch')

    # path-settings
    parser.add_argument('-d', '--dataset', default="flickr25k",
                        help='Dataset name, options: cifar-10, nus-wide-tc-10(21), flickr25k, coco')
    parser.add_argument('-r', '--root', default="/root/autodl-tmp/datasets",
                        help='Path of dataset')
    parser.add_argument('-C', '--checkpoint', default="/root/autodl-tmp/checkpoints", type=str,
                        help='Path of checkpoint.')

    # Method-settings
    parser.add_argument('-c', '--code_length', default=16, type=int,
                        help='Binary hash code length.(default: 32)')
    parser.add_argument('--distance_measure', default='DU3H',
                        help='method to measure distance between points, options: cos, DU3H')
    parser.add_argument('--loss', default='SSDH',
                        help='loss-functions, options: SSDH, DATE')

    # running-settings
    parser.add_argument('-T', '--max-iter', default=60, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-3)')
    parser.add_argument('-q', '--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('-t', '--num-train', default=5000, type=int,
                        help='Number of training data points.(default: 5000)')
    parser.add_argument('-w', '--num-workers', default=4, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='Batch size.(default: 24)')
    parser.add_argument('-a', '--arch', default='vgg16', type=str,
                        help='CNN architecture.(default: vgg16)')
    parser.add_argument('-k', '--topk', default=5000, type=int,
                        help='Calculate map of top k.(default: 5000)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print log.')
    parser.add_argument('--mode', default='train',
                        help='running mode, options: train, resume, evaluate')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('-e', '--evaluate-interval', default=500, type=int,
                        help='Interval of evaluation.(default: 500)')
    parser.add_argument('-s', '--snapshot-interval', default=800, type=int,
                        help='Interval of evaluation.(default: 800)')
    parser.add_argument('--alpha', default=2, type=float,
                        help='Hyper-parameter.(default:2)')
    parser.add_argument('--beta', default=2, type=float,
                        help='Hyper-parameter.(default:2)')

    args = parser.parse_args()

    # 存储模型文件的名字
    args.t_name = f"{args.dataset}_HashLen-{args.code_length}_dist-{args.distance_measure}.t"

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


if __name__ == '__main__':
    run()
