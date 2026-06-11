import numpy as np
import torch

def export_to_bvh(original_bvh_path, output_bvh_path, motion_tensor, fps=60):
    """
    Takes the generated PyTorch tensor and writes it into a standard .bvh file.
    motion_tensor: [3, Time, Joints] 
    """
    # 1. Convert tensor back to numpy [Time, Joints, Channels(3)]
    motion_array = motion_tensor.permute(1, 2, 0).detach().cpu().numpy()
    num_frames = motion_array.shape[0]
    
    # Flatten the rotations so each frame is one long row of floats
    motion_flattened = motion_array.reshape(num_frames, -1)

    # 2. Extract the Hierarchy from the original file
    with open(original_bvh_path, 'r') as f:
        lines = f.readlines()
        
    hierarchy_lines = []
    for line in lines:
        hierarchy_lines.append(line)
        if line.strip() == "MOTION":
            break # Stop copying once we hit the motion block
            
    # 3. Write the new .bvh file
    with open(output_bvh_path, 'w') as f:
        # Write the Skeleton
        f.writelines(hierarchy_lines)
        
        # Write the Motion Header
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")
        
        # Write the Generated Motion Data
        # Note: We append 3 zeros to the start of each row to account for the Root (Hips) Position,
        # which our model did not predict (we only predicted rotations).
        for frame in motion_flattened:
            root_position = "0.000000 0.000000 0.000000 "
            rotations = " ".join([f"{val:.6f}" for val in frame])
            f.write(root_position + rotations + "\n")
            
    print(f"Successfully exported stylized animation to: {output_bvh_path}")