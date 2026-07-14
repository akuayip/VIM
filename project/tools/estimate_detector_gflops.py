"""
Estimate GFLOPs for the Vim-Det Faster R-CNN detector.

This is a transparent estimator for the Vim/Mamba detector path when fvcore
tracing fails on custom Triton ops. It combines the theoretical Vim SSM core
complexity from the Vision Mamba paper with static estimates for the FPN, RPN,
and ROI box head used in configs/models/faster_rcnn_vimdet.py.

The result is an estimate, not a profiler measurement. ROI costs depend on the
number of proposals, so keep --num-rois explicit in reports.
"""

import argparse
from dataclasses import dataclass


def conv2d_flops(h, w, cin, cout, kernel, groups=1):
    return h * w * cout * (cin // groups) * kernel * kernel


def conv_transpose2d_flops(h, w, cin, cout, kernel, stride):
    out_h = h * stride
    out_w = w * stride
    return out_h * out_w * cout * cin * kernel * kernel


def linear_flops(in_features, out_features, batch=1):
    return batch * in_features * out_features


@dataclass
class Estimate:
    name: str
    flops: float
    note: str = ""

    @property
    def gflops(self):
        return self.flops / 1e9


def estimate_vim_ssm_core(input_size, patch_size, embed_dim, depth, ssm_state):
    tokens_per_side = input_size // patch_size
    num_tokens = tokens_per_side * tokens_per_side
    flops = 8 * num_tokens * embed_dim * ssm_state * depth
    return Estimate(
        "Vim SSM core",
        flops,
        f"8*M*D*N*L, M={num_tokens}, D={embed_dim}, N={ssm_state}, L={depth}",
    )


def estimate_patch_embed(input_size, patch_size, embed_dim):
    h = input_size // patch_size
    w = input_size // patch_size
    flops = conv2d_flops(h, w, 3, embed_dim, patch_size)
    return Estimate("Patch embedding", flops, f"Conv2d 3->{embed_dim}, k={patch_size}, output={h}x{w}")


def estimate_simple_feature_pyramid(input_size, patch_size, embed_dim, out_channels):
    base = input_size // patch_size
    items = []

    # simfp_2: ConvTranspose2d 192->96 stride2, ConvTranspose2d 96->48 stride2,
    #          Conv2d 48->256 k1, Conv2d 256->256 k3 at 256x256.
    f = 0
    f += conv_transpose2d_flops(base, base, embed_dim, embed_dim // 2, 2, 2)
    f += conv_transpose2d_flops(base * 2, base * 2, embed_dim // 2, embed_dim // 4, 2, 2)
    f += conv2d_flops(base * 4, base * 4, embed_dim // 4, out_channels, 1)
    f += conv2d_flops(base * 4, base * 4, out_channels, out_channels, 3)
    items.append(Estimate("SimpleFeaturePyramid p2", f, f"output={base*4}x{base*4}"))

    # simfp_3: ConvTranspose2d 192->96 stride2, Conv2d 96->256 k1,
    #          Conv2d 256->256 k3 at 128x128.
    f = 0
    f += conv_transpose2d_flops(base, base, embed_dim, embed_dim // 2, 2, 2)
    f += conv2d_flops(base * 2, base * 2, embed_dim // 2, out_channels, 1)
    f += conv2d_flops(base * 2, base * 2, out_channels, out_channels, 3)
    items.append(Estimate("SimpleFeaturePyramid p3", f, f"output={base*2}x{base*2}"))

    # simfp_4: Conv2d 192->256 k1, Conv2d 256->256 k3 at 64x64.
    f = 0
    f += conv2d_flops(base, base, embed_dim, out_channels, 1)
    f += conv2d_flops(base, base, out_channels, out_channels, 3)
    items.append(Estimate("SimpleFeaturePyramid p4", f, f"output={base}x{base}"))

    # simfp_5: MaxPool stride2, Conv2d 192->256 k1, Conv2d 256->256 k3 at 32x32.
    f = 0
    f += conv2d_flops(base // 2, base // 2, embed_dim, out_channels, 1)
    f += conv2d_flops(base // 2, base // 2, out_channels, out_channels, 3)
    items.append(Estimate("SimpleFeaturePyramid p5", f, f"output={base//2}x{base//2}"))

    return items


def estimate_rpn(input_size, out_channels, num_anchors):
    items = []
    total = 0
    for stride in (4, 8, 16, 32, 64):
        h = input_size // stride
        w = input_size // stride
        f = 0
        f += conv2d_flops(h, w, out_channels, out_channels, 3)
        f += conv2d_flops(h, w, out_channels, num_anchors, 1)
        f += conv2d_flops(h, w, out_channels, num_anchors * 4, 1)
        total += f
    items.append(Estimate("RPN head p2-p6", total, "shared 3x3 conv + objectness + bbox deltas"))
    return items


def estimate_roi_box_head(num_rois, out_channels, pooler_resolution, num_classes):
    h = pooler_resolution
    w = pooler_resolution
    f = 0
    for _ in range(4):
        f += num_rois * conv2d_flops(h, w, out_channels, out_channels, 3)
    f += linear_flops(out_channels * h * w, 1024, num_rois)
    f += linear_flops(1024, num_classes + 1, num_rois)
    f += linear_flops(1024, num_classes * 4, num_rois)
    return [
        Estimate(
            "ROI box head",
            f,
            f"assumes {num_rois} ROIs/image, 4 convs, 1 fc, {num_classes} foreground class",
        )
    ]


def count_params_from_config(embed_dim, depth, out_channels, num_classes):
    # This script reports params from the measured analyze_model output better
    # than re-estimating every Mamba parameter. Keep this as context only.
    return {
        "expected_model_params_m": 26.2
        if (embed_dim, depth, out_channels, num_classes) == (192, 24, 256, 1)
        else None
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate full detector GFLOPs for Vim-Det.")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--ssm-state", type=int, default=16)
    parser.add_argument("--out-channels", type=int, default=256)
    parser.add_argument("--num-anchors", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-rois", type=int, default=1000)
    parser.add_argument("--pooler-resolution", type=int, default=7)
    args = parser.parse_args()

    estimates = [
        estimate_vim_ssm_core(args.input_size, args.patch_size, args.embed_dim, args.depth, args.ssm_state),
        estimate_patch_embed(args.input_size, args.patch_size, args.embed_dim),
    ]
    estimates += estimate_simple_feature_pyramid(args.input_size, args.patch_size, args.embed_dim, args.out_channels)
    estimates += estimate_rpn(args.input_size, args.out_channels, args.num_anchors)
    estimates += estimate_roi_box_head(
        args.num_rois, args.out_channels, args.pooler_resolution, args.num_classes
    )

    total = sum(item.flops for item in estimates)

    print("\nEstimated Vim-Det detector GFLOPs")
    print("=" * 72)
    print(f"Input size     : {args.input_size}x{args.input_size}")
    print(f"Patch size     : {args.patch_size}")
    print(f"Vim config     : D={args.embed_dim}, L={args.depth}, SSM N={args.ssm_state}")
    print(f"ROI assumption : {args.num_rois} proposals/image")
    print("-" * 72)
    for item in estimates:
        print(f"{item.name:<30} {item.gflops:>10.3f} GFLOPs  {item.note}")
    print("-" * 72)
    print(f"{'Total estimate':<30} {total / 1e9:>10.3f} GFLOPs")

    params = count_params_from_config(args.embed_dim, args.depth, args.out_channels, args.num_classes)
    if params["expected_model_params_m"] is not None:
        print(f"Measured params : {params['expected_model_params_m']:.1f}M from analyze_model.py")

    print("\nNotes:")
    print("- FLOPs use one multiply-add as one operation, matching common fvcore-style reporting.")
    print("- Vim SSM is the theoretical core term from the Vision Mamba paper: 8*M*D*N*L.")
    print("- This excludes post-processing/NMS and treats ROI count as fixed; report --num-rois with the number.")


if __name__ == "__main__":
    main()
