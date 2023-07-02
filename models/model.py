import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Space_Proj(nn.Module):
    def __init__(self, in_dim, inner_dim=256, out_dim=256):
        super(Space_Proj, self).__init__()

        self.block = ProjBlock(in_dim, inner_dim, out_dim)

    def forward(self, x):
        x = self.block(x)
        return x


class ProjBlock(nn.Module):
    def __init__(self, in_dim, inner_dim=256, out_dim=256):
        super(ProjBlock, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_dim, inner_dim, 1, 1, dilation=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.conv2 =  nn.Conv2d(inner_dim, inner_dim, 1, 1, dilation=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(inner_dim)
        self.conv3 = nn.Conv2d(inner_dim, out_dim, 1, 1, dilation=1, padding=0, bias=True)

    def forward(self, x):
        residual = self.relu(self.bn1(self.conv1(x)))

        out = self.conv2(residual) + residual
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        
        return out


class Flow(nn.Module):
    def __init__(self, in_dim, inner_dim=256, out_dim=2):
        super(Flow, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inner_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, q, k):
        x = torch.cat((q, k), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        return x


class HVC(nn.Module):
    def __init__(self, base_encoder, args):
        super(HVC, self).__init__()

        self.momentum = args.momentum_target
        self.pos_radius = args.pos_radius
        self.total_steps = int(args.num_steps * args.epochs)
        self.current_step = 0

        self.online_encoder = base_encoder(head_type='early_return')
        self.target_encoder = base_encoder(head_type='early_return')
        self.remove_strides(self.online_encoder, args.remove_stride_layers)
        self.remove_strides(self.target_encoder, args.remove_stride_layers)

        hidden_dim, _ = self.infer_dims(args.img_size)

        self.online_space_projector = Space_Proj(in_dim=hidden_dim, 
                                                inner_dim=args.projection_hidden_dim,
                                                out_dim=args.projection_dim)
        self.target_space_projector = Space_Proj(in_dim=hidden_dim, 
                                                inner_dim=args.projection_hidden_dim,
                                                out_dim=args.projection_dim)
        self.space_predictor = Space_Proj(in_dim=args.projection_dim, 
                                          inner_dim=args.prediction_hidden_dim,
                                          out_dim=args.projection_dim)

        self.flow = Flow(in_dim=args.projection_dim * 2, inner_dim=args.prediction_hidden_dim, out_dim=2)

        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        for param_o, param_t in zip(self.online_space_projector.parameters(), self.target_space_projector.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def remove_strides(self, model, layer_names):
        """reduce the stride of some residual blocks"""
        for layer in layer_names:
            for m in getattr(model, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = tuple(1 for _ in m.stride)

    def infer_dims(self, img_size):
        dummy_input = torch.zeros(1, 3, img_size, img_size).to(next(self.online_encoder.parameters()).device)
        dummy_out = self.online_encoder(dummy_input)
        return dummy_out.shape[1], dummy_out.shape[-1]

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Momentum update of the momentum encoder"""
        _contrast_momentum = 1. - (1. - self.momentum) * (np.cos(np.pi * self.current_step / self.total_steps) + 1) / 2.
        self.current_step = self.current_step + 1
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * _contrast_momentum + param_o.data * (1. - _contrast_momentum)
        for param_o, param_t in zip(self.online_space_projector.parameters(), self.target_space_projector.parameters()):
            param_t.data = param_t.data * _contrast_momentum + param_o.data * (1. - _contrast_momentum)

    def hybrid_loss(self, q, k, coord_q, coord_k, pos_radius=0.5):
        """ q, k: N * C * H * W
            coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
            Following PixPro: https://github.com/zdaxie/PixPro/
        """
        N, C, H, W = q.shape
        # generate center_coord, width, height
        # [1, 32, 32]
        x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
        y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)

        # [bs, 1, 1]
        q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
        q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
        k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
        k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
        # [bs, 1, 1]
        q_start_x = coord_q[:, 0].view(-1, 1, 1)
        q_start_y = coord_q[:, 1].view(-1, 1, 1)
        k_start_x = coord_k[:, 0].view(-1, 1, 1)
        k_start_y = coord_k[:, 1].view(-1, 1, 1)

        # [bs, 1, 1]
        q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
        k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
        max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

        # [bs, 32, 32]
        center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

        # [bs, 1024, 1024]
        dist_center = torch.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                                 + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
        # [bs, 1024, 1024]
        pos_mask = (dist_center < pos_radius).float().detach()

        # spatial loss
        # [bs, 1024, 1024]
        spatial_logit = torch.bmm(q.view(N, C, -1).transpose(1, 2), k.view(N, C, -1))
        spatial_loss = (spatial_logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

        # temporal loss
        # [bs, 2, 32, 32]
        flow_f = F.normalize(self.flow(q, k), dim=1)
        flow_b = F.normalize(self.flow(k, q), dim=1)
        temporal_logit = torch.bmm(flow_f.view(N, 2, -1).transpose(1, 2), flow_b.view(N, 2, -1))
        temporal_loss = (temporal_logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

        return -2 * (spatial_loss.mean() + temporal_loss.mean())
    
    def forward(self, vid1, vid2, coord1, coord2):
        """
        Input:
            vid1: first views of video clips
            vid2: second views of video clips
            coord1: crop coordinates of vid1
            coord1: crop coordinates of vid2
        Output:
            loss
        """
        # as conventional image dataset
        B, T, C, H, W = vid1.shape
        im_1 = vid1.view(B * T, C, H, W)
        im_2 = vid2.view(B * T, C, H, W)
        coord1 = coord1.view(B * T, -1)
        coord2 = coord2.view(B * T, -1)

        feat_1 = self.online_encoder(im_1)                        # (bs, 512, 32, 32)
        proj_1 = self.online_space_projector(feat_1)
        pred_1 = self.space_predictor(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)                       # (bs, 256, 32, 32)

        feat_2 = self.online_encoder(im_2)
        proj_2 = self.online_space_projector(feat_2)
        pred_2 = self.space_predictor(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)
 
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder()

            feat_1_tar = self.target_encoder(im_1)
            proj_1_tar = self.target_space_projector(feat_1_tar)
            proj_1_tar = F.normalize(proj_1_tar, dim=1)

            feat_2_tar = self.target_encoder(im_2)
            proj_2_tar = self.target_space_projector(feat_2_tar)
            proj_2_tar = F.normalize(proj_2_tar, dim=1)

        loss = self.hybrid_loss(pred_1, proj_2_tar, coord1, coord2, self.pos_radius) + \
               self.hybrid_loss(pred_2, proj_1_tar, coord2, coord1, self.pos_radius)
        
        return loss
