from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import numpy as np
import torch
import re

class BVHMotionParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.joints = []
        self.edges = []
        self.frames = []
        self.frame_time = 0.016667

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


            elif is_motion:
                if line.startswith("Frames:"):
                    pass
                elif line.startswith("Frame Time:"):
                    self.frame_time = float(line.split(":")[1].strip())
                else:

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



        rotations_only = self.frames[:, 3:]


        reshaped = rotations_only.reshape(num_frames, num_joints, 3)

        tensor = torch.tensor(reshaped, dtype=torch.float32)


        return tensor.permute(2, 0, 1)


if __name__ == "__main__":

    test_file = str(get_path("EMO_ANIM_SAMPLE_BVH", "data/sample.bvh"))

    parser = BVHMotionParser(test_file)
    adj_matrix = parser.get_adjacency_matrix()
    motion_tensor = parser.get_motion_tensor()

    print(f"\nFinal Adjacency Matrix Shape: {adj_matrix.shape}")
    print(f"Final Motion Tensor Shape: {motion_tensor.shape} -> [Channels, Time, Joints]")
