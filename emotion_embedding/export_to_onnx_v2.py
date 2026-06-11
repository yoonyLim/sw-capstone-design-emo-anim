from pathlib import Path
import argparse

import torch
import torch.nn as nn

from bvh_loader import BVHMotionParser
from content_encoder import ContentEncoder
from motion_generator import MotionGenerator


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SAMPLE_BVH = (
    r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1"
    r"\beat_english_v0.2.1\1\1_wayne_0_1_1.bvh"
)


class StylizationPipeline(nn.Module):
    """
    UE-facing wrapper around the trained content encoder and generator.

    Input and output tensors are normalized motion tensors in the same layout
    used by bvh_loader.py:

        [batch, channels XYZ, time, joints]

    The anchor mask lets the exported ONNX reproduce the Python inference
    post-processing inside the graph. Root anchoring keeps Hips rotation from
    the neutral input. Finger anchoring keeps generated finger channels from
    destabilizing Manny retargeting while preserving wrist motion.
    """

    def __init__(self, encoder, generator, anchor_mask):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.register_buffer("anchor_mask", anchor_mask)

    def forward(self, normalized_motion, emotion_vector, adj_matrix, style_multiplier):
        style_scale = style_multiplier.reshape(-1, 1)
        modified_emotion = emotion_vector * style_scale

        content_code = self.encoder(normalized_motion, adj_matrix)
        stylized_motion = self.generator(content_code, modified_emotion, adj_matrix)

        return stylized_motion * (1.0 - self.anchor_mask) + normalized_motion * self.anchor_mask


def build_anchor_mask(joint_names, anchor_root=True, anchor_fingers=True):
    mask = torch.zeros(1, 1, 1, len(joint_names), dtype=torch.float32)

    if anchor_root:
        mask[..., 0] = 1.0

    if anchor_fingers:
        for joint_index, joint_name in enumerate(joint_names):
            is_wrist = joint_name in {"RightHand", "LeftHand"}
            is_finger = "Hand" in joint_name and not is_wrist
            if is_finger:
                mask[..., joint_index] = 1.0

    anchored = [joint_names[i] for i in range(len(joint_names)) if mask[0, 0, 0, i] > 0.5]
    print(f"Anchored joints ({len(anchored)}): {', '.join(anchored)}")
    return mask


def load_models(device, num_joints, encoder_weights, generator_weights):
    encoder = ContentEncoder(num_joints=num_joints).to(device)
    encoder.load_state_dict(torch.load(encoder_weights, map_location=device))
    encoder.eval()

    generator = MotionGenerator(num_joints=num_joints).to(device)
    generator.load_state_dict(torch.load(generator_weights, map_location=device))
    generator.eval()

    return encoder, generator


def export_model(
    sample_bvh,
    output_path,
    encoder_weights,
    generator_weights,
    dummy_frames,
    opset,
    anchor_root,
    anchor_fingers,
):
    device = torch.device("cpu")

    sample_bvh = Path(sample_bvh)
    output_path = Path(output_path)
    encoder_weights = Path(encoder_weights)
    generator_weights = Path(generator_weights)

    print("Parsing source BVH skeleton...")
    parser = BVHMotionParser(str(sample_bvh))
    adj_matrix = parser.get_adjacency_matrix().to(device)
    joint_names = parser.joints
    num_joints = len(joint_names)

    print("Loading trained weights...")
    encoder, generator = load_models(device, num_joints, encoder_weights, generator_weights)

    anchor_mask = build_anchor_mask(
        joint_names,
        anchor_root=anchor_root,
        anchor_fingers=anchor_fingers,
    ).to(device)

    pipeline = StylizationPipeline(encoder, generator, anchor_mask).to(device)
    pipeline.eval()

    dummy_motion = torch.randn(1, 3, dummy_frames, num_joints, dtype=torch.float32, device=device)
    dummy_emotion = torch.randn(1, 64, dtype=torch.float32, device=device)
    dummy_multiplier = torch.ones(1, dtype=torch.float32, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX to: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            pipeline,
            (dummy_motion, dummy_emotion, adj_matrix, dummy_multiplier),
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[
                "NeutralMotion_Norm",
                "EmotionVector",
                "AdjacencyMatrix",
                "StyleMultiplier",
            ],
            output_names=["StylizedMotion_Norm"],
            dynamic_axes={
                "NeutralMotion_Norm": {0: "batch_size", 2: "time"},
                "EmotionVector": {0: "batch_size"},
                "StyleMultiplier": {0: "batch_size"},
                "StylizedMotion_Norm": {0: "batch_size", 2: "time"},
            },
        )

    print("SUCCESS: ONNX export complete.")
    print("Inputs: NeutralMotion_Norm [1,3,T,75], EmotionVector [1,64], AdjacencyMatrix [75,75], StyleMultiplier [1]")
    print("Output: StylizedMotion_Norm [1,3,T,75]")


def parse_args():
    parser = argparse.ArgumentParser(description="Export the emo-anim motion stylizer to ONNX for UE5 NNE.")
    parser.add_argument("--sample-bvh", default=DEFAULT_SAMPLE_BVH)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "MotionStylizer.onnx"))
    parser.add_argument("--encoder-weights", default=str(SCRIPT_DIR / "content_encoder_epoch_20.pth"))
    parser.add_argument("--generator-weights", default=str(SCRIPT_DIR / "motion_generator_epoch_20.pth"))
    parser.add_argument("--dummy-frames", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-root-anchor", action="store_true")
    parser.add_argument("--no-finger-anchor", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(
        sample_bvh=args.sample_bvh,
        output_path=args.output,
        encoder_weights=args.encoder_weights,
        generator_weights=args.generator_weights,
        dummy_frames=args.dummy_frames,
        opset=args.opset,
        anchor_root=not args.no_root_anchor,
        anchor_fingers=not args.no_finger_anchor,
    )
