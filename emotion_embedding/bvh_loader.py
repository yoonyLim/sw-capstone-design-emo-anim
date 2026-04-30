import numpy as np
import torch
import re

class BVHMotionParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.joints = []           # List of joint names
        self.edges = []            # (Parent, Child) index pairs
        self.frames = []           # The raw Euler angle motion data
        self.frame_time = 0.016667 # Default 60fps
        
        self._parse_bvh()

    def _parse_bvh(self):
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        is_hierarchy = False
        is_motion = False
        joint_stack = []
        current_joint_idx = -1

        print(f"Parsing BVH: {self.file_path.split('/')[-1]}")

        for line in lines:
            line = line.strip()
            if not line: continue

            if line == "HIERARCHY":
                is_hierarchy = True
                continue
            elif line == "MOTION":
                is_hierarchy = False
                is_motion = True
                continue

            # --- 1. PARSE THE SKELETON TREE ---
            if is_hierarchy:
                if line.startswith("ROOT") or line.startswith("JOINT"):
                    joint_name = line.split()[1]
                    self.joints.append(joint_name)
                    current_joint_idx = len(self.joints) - 1
                    
                    if len(joint_stack) > 0:
                        parent_idx = joint_stack[-1]
                        self.edges.append((parent_idx, current_joint_idx))
                
                elif line == "{":
                    joint_stack.append(current_joint_idx)
                
                elif line == "}":
                    joint_stack.pop()
                    if len(joint_stack) > 0:
                        current_joint_idx = joint_stack[-1]

            # --- 2. PARSE THE MOTION DATA ---
            elif is_motion:
                if line.startswith("Frames:"):
                    pass # We dynamically size the tensor later
                elif line.startswith("Frame Time:"):
                    self.frame_time = float(line.split(":")[1].strip())
                else:
                    # These are the raw floats. We convert them to a numpy array.
                    values = np.array([float(x) for x in line.split()])
                    self.frames.append(values)

        self.frames = np.array(self.frames)
        print(f"Extraction Complete! Found {len(self.joints)} joints and {self.frames.shape[0]} frames.")

    def get_adjacency_matrix(self):
        """Dynamically generates the ST-GCN graph matrix from the parsed edges."""
        num_nodes = len(self.joints)
        A = np.zeros((num_nodes, num_nodes))
        
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1 
            
        A = A + np.eye(num_nodes)
        
        rowsum = A.sum(1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        A_normalized = np.dot(np.dot(d_mat_inv_sqrt, A), d_mat_inv_sqrt)
        return torch.tensor(A_normalized, dtype=torch.float32)

    def get_motion_tensor(self):
        """
        Formats the motion data for the ST-GCN.
        BVH files store 3 rotation channels (X, Y, Z) per joint, plus 3 position channels for the Root.
        We isolate the rotations and reshape to [Channels, Time, Joints].
        """
        num_frames = self.frames.shape[0]
        num_joints = len(self.joints)
        
        # The first 3 columns are Hips X/Y/Z Position. We slice them off to keep only rotations.
        # This assumes exactly 3 rotation channels per joint.
        rotations_only = self.frames[:, 3:] 
        
        # Reshape to [Time, Joints, Channels (X,Y,Z)]
        reshaped = rotations_only.reshape(num_frames, num_joints, 3)
        
        tensor = torch.tensor(reshaped, dtype=torch.float32)
        
        # Permute to the standard PyTorch format: [Channels(3), Time, Joints]
        return tensor.permute(2, 0, 1)


if __name__ == "__main__":
    # Point this to the specific file you just uploaded
    test_file = r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_1_1.bvh" 
    
    parser = BVHMotionParser(test_file)
    adj_matrix = parser.get_adjacency_matrix()
    motion_tensor = parser.get_motion_tensor()
    
    print(f"\nFinal Adjacency Matrix Shape: {adj_matrix.shape}")
    print(f"Final Motion Tensor Shape: {motion_tensor.shape} -> [Channels, Time, Joints]")