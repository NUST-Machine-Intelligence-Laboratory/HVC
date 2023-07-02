import os.path as osp

import torch
import torch.nn as nn

from models import resnet


class From3D(nn.Module):
    ''' Use a 2D convnet as a 3D convnet '''

    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet

    def forward(self, x):
        N, C, T, h, w = x.shape
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w)
        m = self.model(xx)

        return m.view(N, T, *m.shape[-3:]).permute(0, 2, 1, 3, 4)


def infer_dims(model):
    in_sz = 256
    dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(model.parameters()).device)
    dummy_out = model(dummy)
    map_scale = in_sz // dummy_out.shape[-1]
    return (map_scale, map_scale)


def partial_load(pretrained_dict, model, skip_keys=[]):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if
                     k in model_dict and not any([sk in k for sk in skip_keys])}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)


def make_encoder(args, model_type, remove_layers=['layer4']):
    if model_type == 'random18':
        net = resnet.resnet18(pretrained=False)

    elif model_type == 'random50':
        net = resnet.resnet50(pretrained=False)

    elif model_type == 'imagenet18':
        net = resnet.resnet18(pretrained=True)

    elif model_type == 'imagenet50':
        net = resnet.resnet50(pretrained=True)
        
    elif model_type == 'imagenet101':
        net = resnet.resnet101(pretrained=True)

    elif model_type == 'mocov1':
        net = resnet.resnet50(pretrained=False)
        pretrained_path = osp.join('./checkpoints', 'moco_v1_200ep_pretrain.pth.tar')
        net_ckpt = torch.load(pretrained_path)
        net_state = {k.replace('module.encoder_q.', ''): v for k, v in net_ckpt['state_dict'].items() if
                     'module.encoder_q' in k}
        del net_state['fc.weight']
        del net_state['fc.bias']
        partial_load(net_state, net)

    elif model_type == 'simsiam':
        net = resnet.resnet50(pretrained=False)
        pretrained_path = args.model_path
        net_ckpt = torch.load(pretrained_path)
        net_state = {k.replace('module.encoder.', ''): v for k, v in net_ckpt['state_dict'].items() if
                     'module.encoder' in k}
        partial_load(net_state, net)

    elif model_type == 'pixpro':
        net = resnet.resnet50(pretrained=False)
        pretrained_path = args.model_path
        net_ckpt = torch.load(pretrained_path)
        net_state = {k.replace('module.encoder.', ''): v for k, v in net_ckpt['model'].items() if
                     'module.encoder.' in k}
        partial_load(net_state, net)

    elif model_type == 'hvc':
        net = resnet.resnet18(pretrained=False)
        pretrained_path = args.model_path
        net_state = torch.load(pretrained_path)
        partial_load(net_state, net)

    elif model_type == 'fine-grained':
        net = resnet.resnet18(pretrained=False)
        pretrained_path = args.model_path
        net_ckpt = torch.load(pretrained_path)
        net_state = {k.replace('online_encoder.', ''): v for k, v in net_ckpt['model'].items() if 'online_encoder.' in k}
        partial_load(net_state, net)
        
    elif model_type == 'crw':
        net = resnet.resnet18(pretrained=False)
        net.modify(padding='reflect')
        pretrained_path = args.model_path
        net_ckpt = torch.load(pretrained_path)
        state = {}
        for k, v in net_ckpt['model'].items():
            if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                state[k.replace('.1.weight', '.weight')] = v
            if 'encoder.model' in k:
                state[k.replace('encoder.model.', '')] = v
            else:
                state[k] = v
        partial_load(state, net, skip_keys=['head'])
        
    elif model_type == 'dul':
        net = resnet.resnet18(pretrained=False)
        pretrained_path = args.model_path
        net_ckpt = torch.load(pretrained_path)
        net_state = {k.replace('fast_net.backbone.', ''): v for k, v in net_ckpt['model'].items() if 'fast_net.backbone' in k}
        partial_load(net_state, net)
        
    elif model_type == 'dino':
        net = resnet.resnet50(pretrained=False)
        pretrained_path = args.model_path
        net_state = torch.load(pretrained_path)
        partial_load(net_state, net)
        
    else:
        raise ValueError('Invalid model_type.')

    net.modify(remove_layers=remove_layers)
    net = From3D(net)

    map_scale = infer_dims(net)   # map_scale = (8, 8)

    return net, map_scale
