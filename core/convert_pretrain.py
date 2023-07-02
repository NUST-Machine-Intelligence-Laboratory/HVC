import torch
from collections import OrderedDict
from argparse import ArgumentParser


def main(args):
    print(f"Loading checkpoint")
    state_dict = torch.load(args.input_dir, map_location='cpu')

    new_ckpt = OrderedDict()


    # baseline
    for k, v in state_dict['model'].items():
        if k.startswith('online_encoder.'):
            new_v = v
            new_k = k.replace('online_encoder.', '')
            new_ckpt[new_k] = new_v
        
    # print(new_ckpt)
    torch.save(new_ckpt, args.save_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default='./pretrain_ckpt/hvc.pth', help='Directory to load the checkpoint')
    parser.add_argument('--save_dir', default='./checkpoints/hvc_coco.pth', help='Directory to save the checkpoint')
    args = parser.parse_args()

    main(args)
