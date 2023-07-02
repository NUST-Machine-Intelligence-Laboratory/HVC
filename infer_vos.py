import os
import sys
import numpy as np
import imageio

import torch.multiprocessing as mp

import torch
import torch.nn.functional as F

from torch.cuda.amp import autocast

from core.config import cfg, get_arguments
from models.encoder import make_encoder
from core.timer import Timer
from core.sys_tools import check_dir
from core.palette_davis import palette as davis_palette

from torch.utils.data import DataLoader
from dataloader.dataloader_infer import DataSeg

from labelprop.common import LabelPropVOS_CRW

# deterministic inference
from torch.backends import cudnn

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True


def mask2rgb(mask, palette):
    mask_rgb = palette(mask)
    mask_rgb = mask_rgb[:,:,:3]
    return mask_rgb

def mask_overlay(mask, image, palette):
    """Creates an overlayed mask visualisation"""
    mask_rgb = mask2rgb(mask, palette)
    return 0.5 * image + 0.5 * mask_rgb

class ResultWriter:
    
    def __init__(self, key, palette, out_path):
        self.key = key
        self.palette = palette
        self.out_path = out_path
        self.verbose = cfg.VERBOSE

    def save(self, frames, masks_pred, masks_conf, masks_gt, flags, fn, seq_name):

        subdir_vos = os.path.join(self.out_path, "{}_vos".format(self.key))
        check_dir(subdir_vos, seq_name)

        subdir_vis = os.path.join(self.out_path, "{}_vis".format(self.key))
        check_dir(subdir_vis, seq_name)

        for frame_id, mask in enumerate(masks_pred.split(1, 0)):

            mask = mask[0].numpy().astype(np.uint8)
            filepath = os.path.join(subdir_vos, seq_name, "{}.png".format(fn[frame_id][0]))

            # saving only every 5th frame
            if flags[frame_id] != 0:
                imageio.imwrite(filepath, mask)

            if self.verbose:
                frame = frames[frame_id].numpy()
                frame = np.transpose(frame, [1,2,0])
                overlay = mask_overlay(mask, frame, self.palette)
                filepath = os.path.join(subdir_vis, seq_name, "{}.png".format(fn[frame_id][0]))
                imageio.imwrite(filepath, (overlay * 255.).astype(np.uint8))


def convert_dict(state_dict):
    new_dict = {}
    for k,v in state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict

def mask2tensor(mask, idx, num_classes=cfg.DATASET.NUM_CLASSES):
    h,w = mask.shape
    mask_t = torch.zeros(1,num_classes,h,w)
    mask_t[0, idx] = mask
    return mask_t

def configure_tracks(masks_gt, tracks, num_objects):
    """Selecting masks for initialisation
    Args:
        masks_gt: [T,H,W]
        tracks: [T,2]
    """
    init_masks = {}

    # we always have first mask
    # if there are no instances, it will be simply zero
    H,W = masks_gt[0].shape[-2:]
    init_masks[0] = torch.zeros(1, cfg.DATASET.NUM_CLASSES, H, W)

    for oid in range(cfg.DATASET.NUM_CLASSES):

        t = tracks[oid].item()
        if not t in init_masks:
            init_masks[t] = mask2tensor(masks_gt[oid], oid)
        else:
            init_masks[t] += mask2tensor(masks_gt[oid], oid)

    return init_masks

def make_onehot(mask, HW):
    # convert mask tensor with probabilities to a one-hot tensor
    b,c,h,w = mask.shape

    mask_up = F.interpolate(mask, HW, mode="bilinear", align_corners=True)
    one_hot = torch.zeros_like(mask_up)
    one_hot.scatter_(1, mask_up.argmax(1, keepdim=True), 1)
    one_hot = F.interpolate(one_hot, (h,w), mode="bilinear", align_corners=True)

    return one_hot

def scale_smallest(frame, a):
    H,W = frame.shape[-2:]
    s = a / min(H, W)
    h, w = int(s * H), int(s * W)
    return F.interpolate(frame, (h, w), mode="bilinear", align_corners=True)

def valid_mask(mask):
    """From a tensor [1,C,h,w]
    create [1,C,1,1] 0/1 mask saying which IDs are present"""
    B,C,h,w = mask.shape
    valid = mask.flatten(2,3).sum(-1) > 0
    valid = valid.type_as(mask).view(B,C,1,1)
    return valid

def merge_mask_ids(masks, key0):
    merged_mask = torch.zeros_like(masks[key0])
    for tt, mask in masks.items():
        merged_mask[:,1:] += mask[:,1:]

    probs, ids = merged_mask.max(1, keepdim=True)
    merged_mask.zero_()
    merged_mask.scatter(1, ids, probs)
    merged_mask[:, :1] = 1 - probs
    return merged_mask

def step_seg(cfg, net_hvc, net, labelprop, frames, mask_init):

    # dense tracking: start from the 1st frame
    # keep track of new objects

    T = frames.shape[0]
    frames = frames.cuda()

    # scale smallest
    if cfg.TEST.INPUT_SIZE > 0:
        frames = scale_smallest(frames, cfg.TEST.INPUT_SIZE)

    for t in mask_init.keys():
        mask_init[t] = mask_init[t].cuda()

    H,W = mask_init[0].shape[-2:]

    scale_as = lambda x, y: F.interpolate(x, y.shape[-2:], mode="bilinear", align_corners=True)
    scale = lambda x, hw: F.interpolate(x, hw, mode="bilinear", align_corners=True)

    # context (cxt) will maintain 
    # the reference frame
    ref_embd = {}   # context embeddings
    ref_masks = {}
    ref_valid = {}

    # we will also keep a bunch
    # of previous frames
    prev_embd = None
    prev_masks = None

    all_masks = []
    all_masks_conf = []

    def add_result(mask):
        mask_up = scale(mask, (H, W))
        nxt_masks_conf, nxt_masks_id = mask_up.max(1)
        all_masks.append(nxt_masks_id.cpu())
        all_masks_conf.append(nxt_masks_conf.cpu())

    # initialising
    hvc_feats = net_hvc(frames[:1].unsqueeze(2)).squeeze(2)
    hvc_feats = F.normalize(hvc_feats,dim=1)

    if cfg.TEST.MERGE:
        se_feats = net(frames[:1].unsqueeze(2)).squeeze(2)
        se_feats = F.normalize(se_feats, dim=1)
        embd0 = torch.cat((hvc_feats * cfg.TEST.LAMBDA, se_feats), dim=1)
        embd0 = F.normalize(embd0, dim=1)
    else:
        embd0 = hvc_feats

    mask0 = scale_as(mask_init[0], embd0)
    add_result(mask0)

    ref_embd[0] = {0: embd0}
    ref_masks[0] = {0: mask0}
    ref_valid[0] = valid_mask(mask0) # [x,c,1,1]

    # add this to the reference context
    # if there are objects
    ref_index = []
    if mask_init[0].sum() > 0:
        ref_index = [0]

    print(">", end='')
    for t in range(1, T):
        print(".", end='')
        sys.stdout.flush()

        # next frame
        frames_batch = frames[t:t+1]

        # source forward pass
        nxt_hvc = net_hvc(frames_batch.unsqueeze(2)).squeeze(2)
        nxt_hvc = F.normalize(nxt_hvc, dim=1)

        if cfg.TEST.MERGE:
            nxt_se = net(frames_batch.unsqueeze(2)).squeeze(2)
            nxt_se = F.normalize(nxt_se, dim=1)
            nxt_embd = torch.cat((nxt_hvc * cfg.TEST.LAMBDA, nxt_se), dim=1)
            nxt_embd = F.normalize(nxt_embd, dim=1)
        else:
            nxt_embd = nxt_hvc

        ref_t = [0] if len(ref_index) == 0 else ref_index

        # for each reference mask
        # we will create own context, then
        # propagate the labels and merge the result
        nxt_masks = {}
        for t0 in ref_t:
            cxt_index = labelprop.context_index(t0, t)
            cxt_embd = [ref_embd[t0][j] for j in cxt_index]
            cxt_masks = [ref_masks[t0][j] for j in cxt_index]
            nxt_masks[t0] = labelprop.predict(cxt_embd, cxt_masks, nxt_embd, cxt_index, t)

        # merging all the masks
        nxt_mask = sum([ref_valid[tt] * nxt_masks[tt] for tt in nxt_masks.keys()])
        #nxt_mask = merge_mask_ids(nxt_masks, ref_t[0])

        if t in mask_init: # not t >= 0
            print("Adding GT mask t = ", t)
            # adding the initial mask if just appeared
            mask_init_dn = scale_as(mask_init[t], nxt_embd)
            mask_init_dn_s = mask_init_dn.sum(1, keepdim=True)
            nxt_mask = (1 - mask_init_dn_s) * nxt_mask + mask_init_dn_s * mask_init_dn

            # adding to context
            ref_embd[t] = {}
            ref_masks[t] = {}
            ref_valid[t] = valid_mask(mask_init[t])
            ref_index.append(t)

        add_result(nxt_mask)
        ref_t = [0] if len(ref_index) == 0 else ref_index

        #
        # updating the context
        # two parts: for every initial index, keep first N and last M frames
        for t0 in ref_t:
            ref_embd[t0][t] = nxt_embd.clone()
            ref_masks[t0][t] = nxt_mask.clone()

            index_short = labelprop.context_long(t0, t)

            tsteps = list(ref_embd[t0].keys())
            for tt in tsteps:
                if t - tt > cfg.TEST.CXT_SIZE and not tt in index_short:
                    del ref_embd[t0][tt]
                    del ref_masks[t0][tt]

    print('<')
    masks_pred = torch.cat(all_masks, 0)
    masks_pred_conf = torch.cat(all_masks_conf, 0)

    return masks_pred, masks_pred_conf


if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])  

    # initialising the dirs
    check_dir(args.mask_output_dir, "{}_vis".format(cfg.TEST.EXP))
    check_dir(args.mask_output_dir, "{}_vos".format(cfg.TEST.EXP))


    # Loading the model
    model_hvc, _ = make_encoder(args, model_type=args.test_model)

    # setting the evaluation mode
    model_hvc = model_hvc.cuda()
    model_hvc.eval()
    
    if cfg.TEST.MERGE:
        model, _ = make_encoder(args, model_type=args.merge_model)
        model = model.cuda()
        model.eval()
    else:
        model = None
    
    labelprop = LabelPropVOS_CRW(cfg)

    dataset = DataSeg(cfg, args.infer_list)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, \
                                    drop_last=False)
    palette = dataloader.dataset.get_palette()

    timer = Timer()
    N = len(dataloader)
    pool = mp.Pool(processes=args.pool_workers)
    writer = ResultWriter(cfg.TEST.EXP, davis_palette, args.mask_output_dir)

    for iter, batch in enumerate(dataloader):
        with autocast():
            frames_orig, frames, masks_gt, tracks, num_ids, fns, flags, seq_name = batch

            print("Sequence {:02d} | {}".format(iter, seq_name[0]))

            masks_gt = masks_gt.flatten(0,1)
            frames_orig = frames_orig.flatten(0,1)
            frames = frames.flatten(0,1)
            tracks = tracks.flatten(0,1)
            flags = flags.flatten(0,1)

            init_masks = configure_tracks(masks_gt, tracks, num_ids[0])
            assert 0 in init_masks, "initial frame has no instances"

            with torch.no_grad():
                masks_pred, masks_conf = step_seg(cfg, model_hvc, model, labelprop, frames, init_masks)

            frames_orig = dataset.denorm(frames_orig)

            pool.apply_async(writer.save, args=(frames_orig, masks_pred.cpu(), masks_conf.cpu(), masks_gt.cpu(), flags, fns, seq_name[0]))

    timer.stage("Inference completed")
    pool.close()
    pool.join()
