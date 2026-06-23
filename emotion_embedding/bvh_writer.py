import numpy as np
import torch

def export_to_bvh(original_bvh_path, output_bvh_path, motion_tensor, fps=60):
    """
    Takes the generated PyTorch tensor and writes it into a standard .bvh file.
    motion_tensor: [3, Time, Joints]
    """

    motion_array = motion_tensor.permute(1, 2, 0).detach().cpu().numpy()
    num_frames = motion_array.shape[0]


    motion_flattened = motion_array.reshape(num_frames, -1)


    with open(original_bvh_path, 'r') as f:
        lines = f.readlines()

    hierarchy_lines = []
    for line in lines:
        hierarchy_lines.append(line)
        if line.strip() == "MOTION":
            break


    with open(output_bvh_path, 'w') as f:

        f.writelines(hierarchy_lines)


        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")




        for frame in motion_flattened:
            root_position = "0.000000 0.000000 0.000000 "
            rotations = " ".join([f"{val:.6f}" for val in frame])
            f.write(root_position + rotations + "\n")

    print(f"Successfully exported stylized animation to: {output_bvh_path}")
