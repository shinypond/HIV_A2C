import os
import argparse
import logging

from lib.agent import A2C_AGENT
from lib.utils import dict2namespace, load_config, load_and_copy_config, str2bool


def main(args):
    args.logdir = os.path.join('./logs/', args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    
    if not args.resume:
        config = load_and_copy_config(args.config, os.path.join(args.logdir, 'config_copy.yml'))
    else:
        config = load_config(args.config)

    config.device = f'cuda:{args.gpu}'

    log_file = os.path.join(args.logdir, f'{args.mode}_log.txt')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_file, mode='w')
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel('INFO')

    agent = A2C_AGENT()
    if args.mode == 'train':
        agent.train(config, args.logdir, args.resume)
    elif args.mode == 'eval':
        agent.eval(config, args.logdir, ckpt_num=config.eval.ckpt_num)
    else:
        raise ValueError(f'Mode {args.mode} not recognized')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./lib/configs/config.yml')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)
