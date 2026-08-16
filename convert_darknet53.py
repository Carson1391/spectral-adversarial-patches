"""
Convert darknet53.weights (Darknet binary format) to a PyTorch state_dict .pt file.
Darknet-53 ImageNet pretrained backbone from https://pjreddie.com/media/files/darknet53.weights

Darknet-53 architecture (53 conv layers):
  Stage 0: conv 3->32, 3x3, s1
  Stage 1: conv 32->64, 3x3, s2  + 1x resblock(64)
  Stage 2: conv 64->128, 3x3, s2 + 2x resblock(128)
  Stage 3: conv 128->256, 3x3, s2 + 8x resblock(256)
  Stage 4: conv 256->512, 3x3, s2 + 8x resblock(512)
  Stage 5: conv 512->1024, 3x3, s2 + 4x resblock(1024)
  Head:   conv 1024->1000, 1x1, s1 (no BN, has bias)

  resblock(C) = conv C->2C (3x3, BN) + conv 2C->C (1x1, BN)
  Total convs = 1 + 1+2 + 1+4 + 1+16 + 1+16 + 1+8 + 1 = 53
"""

import struct
import numpy as np
import torch
import os

WEIGHTS_PATH = r"C:\Users\carso\Desktop\modelspace\darknet53.weights"
OUTPUT_PATH  = r"C:\Users\carso\Desktop\modelspace\darknet53.pt"

# Residual block counts per stage: [1, 2, 8, 8, 4]
RESBLOCKS = [1, 2, 8, 8, 4]


def build_conv_layers():
    """Generate the full 53-layer Darknet-53 conv spec."""
    layers = []
    # Stage 0: initial conv
    layers.append((3, 32, 3, 1, True))

    # Stages 1-5: downsample conv + N resblocks
    channels = 32  # current channel count after stage 0
    for stage_idx, n_res in enumerate(RESBLOCKS):
        out_c = channels * 2
        # Downsample conv: channels -> out_c, 3x3, stride 2
        layers.append((channels, out_c, 3, 2, True))
        channels = out_c

        # N residual blocks: each = conv C->C/2 (1x1 reduce) + conv C/2->C (3x3 expand)
        for _ in range(n_res):
            layers.append((channels, channels // 2, 1, 1, True))  # 1x1 reduce
            layers.append((channels // 2, channels, 3, 1, True))   # 3x3 expand

    # Classification head: 1x1 conv to 1000 classes, no BN, has bias
    layers.append((channels, 1000, 1, 1, False))

    assert len(layers) == 53, f"Expected 53 conv layers, got {len(layers)}"
    return layers


def read_floats(f, n):
    """Read n float32 values from binary file, return as numpy array."""
    data = f.read(n * 4)
    return np.frombuffer(data, dtype=np.float32).copy()


def convert():
    conv_layers = build_conv_layers()
    print(f"Darknet-53: {len(conv_layers)} conv layers")

    with open(WEIGHTS_PATH, "rb") as f:
        # Read header: major, minor, revision (3 x int32)
        major = struct.unpack("i", f.read(4))[0]
        minor = struct.unpack("i", f.read(4))[0]
        if major * 10 + minor >= 2:
            revision = struct.unpack("i", f.read(4))[0]
            print(f"Darknet weights version: {major}.{minor}.{revision}")
        # Both old and new format have a size_t "seen" field (8 bytes on 64-bit)
        f.read(8)
        print(f"Skipped 8-byte 'seen' field.")

        state_dict = {}

        for i, (in_c, out_c, k, stride, has_bn) in enumerate(conv_layers):
            # Conv weights: (out_c, in_c, k, k)
            conv_w = read_floats(f, out_c * in_c * k * k)
            conv_w = conv_w.reshape(out_c, in_c, k, k)
            state_dict[f"conv{i}.weight"] = torch.from_numpy(conv_w)

            if has_bn:
                # BN: bias, weight(gamma), running_mean, running_var (each out_c floats)
                bn_bias   = read_floats(f, out_c)
                bn_weight = read_floats(f, out_c)
                bn_mean   = read_floats(f, out_c)
                bn_var    = read_floats(f, out_c)
                state_dict[f"bn{i}.bias"]         = torch.from_numpy(bn_bias)
                state_dict[f"bn{i}.weight"]       = torch.from_numpy(bn_weight)
                state_dict[f"bn{i}.running_mean"] = torch.from_numpy(bn_mean)
                state_dict[f"bn{i}.running_var"]  = torch.from_numpy(bn_var)
                state_dict[f"bn{i}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
            else:
                # No BN: conv has bias (out_c floats)
                conv_bias = read_floats(f, out_c)
                state_dict[f"conv{i}.bias"] = torch.from_numpy(conv_bias)

            tag = "BN" if has_bn else "bias"
            print(f"  Layer {i:2d}: conv{in_c:>4}->{out_c:>4} {k}x{k} s={stride} {tag}  OK")

        # Check if we've consumed all data
        remaining = len(f.read())
        if remaining == 0:
            print("\nAll weights consumed cleanly.")
        else:
            print(f"\nWARNING: {remaining} bytes remaining in file!")

    # Save as PyTorch state_dict
    torch.save(state_dict, OUTPUT_PATH)
    print(f"\nSaved state_dict to: {OUTPUT_PATH}")
    print(f"Total tensors: {len(state_dict)}")
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    convert()
