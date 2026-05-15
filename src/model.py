import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# =========================================================
# Basic block
# =========================================================
class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# Fixed PVTv2-B5 encoder
# =========================================================
class PVTv2B5Encoder(nn.Module):
    """
    PVTv2-B5 backbone wrapper.

    Input:
        RGB image, shape [B, 3, H, W]

    Output:
        [c1, c2, c3, c4], each projected to out_ch channels.
    """

    def __init__(self, in_chans=3, out_ch=256, pretrained=True):
        super().__init__()

        print("Loading PVTv2-B5 backbone...")
        print(f"Pretrained backbone: {pretrained}")

        self.body = timm.create_model(
            "pvt_v2_b5",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_chans,
        )

        feat_channels = self.body.feature_info.channels()
        feat_reductions = self.body.feature_info.reduction()

        if len(feat_channels) != 4:
            raise RuntimeError(
                f"Expected 4 feature stages from pvt_v2_b5, "
                f"but got {len(feat_channels)}: {feat_channels}"
            )

        self.proj_c1 = nn.Conv2d(feat_channels[0], out_ch, kernel_size=1)
        self.proj_c2 = nn.Conv2d(feat_channels[1], out_ch, kernel_size=1)
        self.proj_c3 = nn.Conv2d(feat_channels[2], out_ch, kernel_size=1)
        self.proj_c4 = nn.Conv2d(feat_channels[3], out_ch, kernel_size=1)

        print(f"PVTv2-B5 feature channels: {feat_channels}")
        print(f"PVTv2-B5 feature reductions: {feat_reductions}")

    def forward(self, x):
        feats = self.body(x)

        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 feature maps, but got {len(feats)}")

        c1 = self.proj_c1(feats[0])
        c2 = self.proj_c2(feats[1])
        c3 = self.proj_c3(feats[2])
        c4 = self.proj_c4(feats[3])

        return [c1, c2, c3, c4]


# =========================================================
# FPN neck
# =========================================================
class FPNNeck(nn.Module):
    """
    Top-down feature pyramid fusion.
    """

    def __init__(self, in_channels=256):
        super().__init__()

        self.smooth_c4 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.smooth_c3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.smooth_c2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.smooth_c1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        p4 = c4
        p3 = c3 + F.interpolate(p4, size=c3.shape[2:], mode="bilinear", align_corners=False)
        p2 = c2 + F.interpolate(p3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        p1 = c1 + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        p4 = self.smooth_c4(p4)
        p3 = self.smooth_c3(p3)
        p2 = self.smooth_c2(p2)
        p1 = self.smooth_c1(p1)

        return [p1, p2, p3, p4]


# =========================================================
# RC branch
# =========================================================
class RCBranch(nn.Module):
    """
    Region-coherent branch.

    It captures region-level contextual consistency using residual
    dual-scale anisotropic large-kernel modeling.
    """

    def __init__(self, channels):
        super().__init__()

        self.small_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.large_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(15, 1),
                padding=(7, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f_small = self.small_branch(x)
        f_large = self.large_branch(x)

        f = self.fuse(torch.cat([f_small, f_large], dim=1))
        f = self.out_proj(f)

        return x + f


# =========================================================
# BS branch
# =========================================================
class LaplacianDepthwise(nn.Module):
    """
    Fixed channel-wise Laplacian operator.
    """

    def __init__(self, channels):
        super().__init__()

        kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        kernel = kernel.repeat(channels, 1, 1, 1)

        self.register_buffer("weight", kernel)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=1,
            padding=1,
            groups=self.channels,
        )


class LearnableHighPass(nn.Module):
    """
    Learnable high-pass transformation.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BSBranch(nn.Module):
    """
    Boundary-sensitive branch.

    It combines a fixed Laplacian prior and a learnable high-pass
    transformation to enhance local structural discontinuities.
    """

    def __init__(self, channels):
        super().__init__()

        self.lap = LaplacianDepthwise(channels)
        self.learnable_hp = LearnableHighPass(channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=3, padding=1),
            ConvBNReLU(channels, channels, kernel_size=3, padding=1),
        )

        self.edge_head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        edge_fixed = self.lap(x)
        edge_learn = self.learnable_hp(x)
        edge = edge_fixed + edge_learn

        out = torch.cat([x, edge], dim=1)
        out = self.fuse(out)
        out = self.refine(out)

        edge_logits = self.edge_head(out)

        return out, edge_logits


# =========================================================
# SC function
# =========================================================
class SCfunction(nn.Module):
    """
    Spatial coordination function.

    It predicts coordination logits from:
        shared representation,
        boundary probability map,
        branch discrepancy map.
    """

    def __init__(self, channels):
        super().__init__()

        in_ch = channels + 2
        mid = max(channels // 4, 32)

        self.sc_pred = nn.Sequential(
            ConvBNReLU(in_ch, mid, kernel_size=1, padding=0),
            ConvBNReLU(mid, mid, kernel_size=3, padding=1),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )

    def forward(self, f_base, edge_logits, f_rc, f_bs):
        edge_prob = torch.sigmoid(edge_logits)
        diff_map = torch.mean(torch.abs(f_bs - f_rc), dim=1, keepdim=True)

        sc_input = torch.cat([f_base, edge_prob, diff_map], dim=1)
        coord_logits = self.sc_pred(sc_input)

        return coord_logits


# =========================================================
# DSS block
# =========================================================
class DSSblock(nn.Module):
    """
    Dynamic structural specialization block.

    It contains:
        RCBranch,
        BSBranch,
        SCfunction.
    """

    def __init__(self, channels=256):
        super().__init__()

        self.rc_branch = RCBranch(channels)
        self.bs_branch = BSBranch(channels)
        self.sc_function = SCfunction(channels)

    def forward(self, f_base):
        f_rc = self.rc_branch(f_base)
        f_bs, edge_logits = self.bs_branch(f_base)

        coord_logits = self.sc_function(
            f_base=f_base,
            edge_logits=edge_logits,
            f_rc=f_rc,
            f_bs=f_bs,
        )

        w_bs = torch.sigmoid(coord_logits)
        f_d = w_bs * f_bs + (1.0 - w_bs) * f_rc

        aux = {
            # Training-compatible names
            "router_logits": coord_logits,
            "edge_logits": edge_logits,
            "f_low": f_rc,
            "f_high": f_bs,
            "w_high": w_bs,

            # Paper-friendly names
            "coord_logits": coord_logits,
            "w_bs": w_bs,
            "f_rc": f_rc,
            "f_bs": f_bs,
        }

        return f_d, aux


# =========================================================
# DSS head
# =========================================================
class DSSHead(nn.Module):
    """
    Shared representation construction and dynamic structural specialization.
    """

    def __init__(self, in_channels=256, embedding_dim=256):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.dss_block = DSSblock(channels=embedding_dim)

        self.dropout = nn.Dropout(0.1)
        self.seg_head = nn.Conv2d(embedding_dim, 1, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        size = c1.shape[2:]

        c4_up = F.interpolate(c4, size=size, mode="bilinear", align_corners=False)
        c3_up = F.interpolate(c3, size=size, mode="bilinear", align_corners=False)
        c2_up = F.interpolate(c2, size=size, mode="bilinear", align_corners=False)

        f_base = self.linear_fuse(torch.cat([c4_up, c3_up, c2_up, c1], dim=1))
        f_d, aux = self.dss_block(f_base)

        feat = self.dropout(f_d)
        seg_logits_small = self.seg_head(feat)

        aux["f_base"] = f_base
        aux["f_d"] = f_d

        return seg_logits_small, feat, aux


# =========================================================
# PPR
# =========================================================
class LearnableUp4_SchemeA(nn.Module):
    """
    Coarse full-resolution decoder.
    """

    def __init__(self, feat_ch=256, p1_dim=256, p1_reduce_dim=64, mid_ch=64):
        super().__init__()

        self.reduce_p1 = nn.Sequential(
            nn.Conv2d(p1_dim, p1_reduce_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(p1_reduce_dim),
            nn.ReLU(inplace=True),
        )

        in_ch = feat_ch + 1 + p1_reduce_dim

        self.fuse = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, kernel_size=3, padding=1),
            ConvBNReLU(mid_ch, mid_ch, kernel_size=3, padding=1),
        )

        self.up_pred = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, feat_small, seg_logits_small, p1_feat):
        p1_small = self.reduce_p1(p1_feat)
        p1_to_small = F.interpolate(
            p1_small,
            size=feat_small.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([feat_small, seg_logits_small, p1_to_small], dim=1)
        x = self.fuse(x)
        coarse_full_logits = self.up_pred(x)

        return coarse_full_logits


class RefinementFullOnce(nn.Module):
    """
    Final full-resolution refinement.
    """

    def __init__(self, p1_dim=256, reduce_dim=96):
        super().__init__()

        self.reduce_p1 = nn.Sequential(
            nn.Conv2d(p1_dim, reduce_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
        )

        in_ch = 3 + 1 + reduce_dim + 1 + 1

        self.refine = nn.Sequential(
            ConvBNReLU(in_ch, 64, kernel_size=3, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, rgb, coarse_full_logits, p1_feat, edge_logits, w_bs):
        _, _, H, W = rgb.shape

        p1_reduced = self.reduce_p1(p1_feat)
        p1_full = F.interpolate(
            p1_reduced,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        edge_full = torch.sigmoid(edge_logits)
        edge_full = F.interpolate(
            edge_full,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        w_bs_full = F.interpolate(
            w_bs,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        x = torch.cat(
            [
                rgb,
                coarse_full_logits,
                p1_full,
                edge_full,
                w_bs_full,
            ],
            dim=1,
        )

        residual = self.refine(x)
        final_logits = coarse_full_logits + residual

        return final_logits


# =========================================================
# DSS-USOD
# =========================================================
class DSSUSOD(nn.Module):
    """
    RGB-only DSS-USOD with fixed PVTv2-B5 backbone.

    Output:
        final_logits, coarse_full_logits, seg_logits_small, aux
    """

    def __init__(self, in_chans=3, pretrained_backbone=True, embedding_dim=256):
        super().__init__()

        self.encoder = PVTv2B5Encoder(
            in_chans=in_chans,
            out_ch=256,
            pretrained=pretrained_backbone,
        )

        self.neck = FPNNeck(in_channels=256)

        self.decoder = DSSHead(
            in_channels=256,
            embedding_dim=embedding_dim,
        )

        self.up4 = LearnableUp4_SchemeA(
            feat_ch=embedding_dim,
            p1_dim=256,
            p1_reduce_dim=64,
            mid_ch=64,
        )

        self.refine = RefinementFullOnce(
            p1_dim=256,
            reduce_dim=96,
        )

    def forward(self, x_rgb):
        feats = self.encoder(x_rgb)
        fpn_feats = self.neck(feats)

        seg_logits_small, feat_small, aux = self.decoder(fpn_feats)

        coarse_full_logits = self.up4(
            feat_small,
            seg_logits_small,
            fpn_feats[0],
        )

        final_logits = self.refine(
            x_rgb,
            coarse_full_logits,
            fpn_feats[0],
            aux["edge_logits"],
            aux["w_bs"],
        )

        return final_logits, coarse_full_logits, seg_logits_small, aux


if __name__ == "__main__":
    model = DSSUSOD(in_chans=3, pretrained_backbone=False)
    model.eval()

    x_rgb = torch.randn(2, 3, 352, 352)

    with torch.no_grad():
        final_logits, coarse_full_logits, seg_logits_small, aux = model(x_rgb)

    print("final_logits:", final_logits.shape)
    print("coarse_full_logits:", coarse_full_logits.shape)
    print("seg_logits_small:", seg_logits_small.shape)
    print("edge_logits:", aux["edge_logits"].shape)
    print("coord_logits:", aux["coord_logits"].shape)
    print("f_rc:", aux["f_rc"].shape)
    print("f_bs:", aux["f_bs"].shape)

