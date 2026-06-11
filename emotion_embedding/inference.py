# python inference.py --emotion-source sliders --happiness 1.0 --no-finger-anchor
# python inference.py --emotion-source sliders --happiness 1.0 --no-root-anchor
# python inference.py --emotion-source sliders --happiness 1.0 --output D:\Capstone_Project\Test_Data\happy_test.bvh
# python inference.py --emotion-source sliders --sadness 0.8 --fear 0.4 --output D:\Capstone_Project\Test_Data\sad_fear_test.bvh
# python inference.py --emotion-source sliders --anger 0.7 --surprise 0.5 --style-multiplier 1.2

from pathlib import Path
import argparse
import json
import math
import sys

import torch
import torchaudio

from bvh_loader import BVHMotionParser
from bvh_writer import export_to_bvh
from content_encoder import ContentEncoder
from motion_generator import MotionGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
AUDIO_EMOTION_DIR = PROJECT_DIR / "audio_emotion_vector_extraction"
sys.path.append(str(AUDIO_EMOTION_DIR))

from ast_model import ASTEmotionExtractor


DEFAULT_AUDIO_FILE = r"D:\Capstone_Project\Test_Data\arthur.wav"
DEFAULT_NEUTRAL_BVH = (
    r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1"
    r"\beat_english_v0.2.1\1\1_wayne_0_1_1.bvh"
)
DEFAULT_OUTPUT_FILE = r"D:\Capstone_Project\Test_Data\test0605.bvh"
DEFAULT_RAW_ANCHORS_FILE = AUDIO_EMOTION_DIR / "master_emotion_anchors_raw.json"
DEFAULT_NORMALIZED_ANCHORS_FILE = AUDIO_EMOTION_DIR / "master_emotion_anchors.json"
DEFAULT_ANCHORS_FILE = (
    DEFAULT_RAW_ANCHORS_FILE
    if DEFAULT_RAW_ANCHORS_FILE.exists()
    else DEFAULT_NORMALIZED_ANCHORS_FILE
)

EMOTION_NAMES = (
    "Happiness",
    "Anger",
    "Sadness",
    "Contempt",
    "Surprise",
    "Fear",
    "Disgust",
)


def zero_to_one(value):
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("emotion values must be between 0 and 1")
    return parsed


def resolve_existing_path(path_value, description):
    path = Path(path_value)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_output_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_audio_waveform(audio_path, device):
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    target_audio_len = 160000
    if waveform.shape[1] < target_audio_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_audio_len - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_audio_len]

    return waveform.unsqueeze(0).to(device)


def load_audio_emotion_vector(audio_path, extractor_weights, device):
    print("Loading AST emotion extractor...")
    extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    extractor.load_state_dict(torch.load(extractor_weights, map_location=device))
    extractor.eval()

    waveform = load_audio_waveform(audio_path, device)
    with torch.no_grad():
        return extractor(waveform)


def load_emotion_anchors(anchors_path, device):
    with open(anchors_path, "r", encoding="utf-8") as handle:
        raw_anchors = json.load(handle)

    required_names = ("Neutral",) + EMOTION_NAMES
    missing = [name for name in required_names if name not in raw_anchors]
    if missing:
        raise KeyError(f"Missing emotion anchors in {anchors_path}: {', '.join(missing)}")

    anchors = {}
    for name in required_names:
        vector = torch.tensor(raw_anchors[name], dtype=torch.float32, device=device).reshape(1, -1)
        if vector.shape[1] != 64:
            raise ValueError(f"{name} anchor must have 64 values, but found {vector.shape[1]}")
        anchors[name] = vector

    neutral_l2 = torch.linalg.vector_norm(anchors["Neutral"]).item()
    if 0.95 <= neutral_l2 <= 1.05:
        print(
            "WARNING: Emotion anchors look L2-normalized. "
            "That is useful for cosine emotion classification, but the motion generator "
            "was trained on raw AST vectors. Prefer raw, unnormalized motion anchors."
        )

    return anchors


def build_slider_delta(anchors, intensities):
    neutral = anchors["Neutral"]
    delta = torch.zeros_like(neutral)

    for emotion_name, intensity in intensities.items():
        if intensity <= 0.0:
            continue
        delta += (anchors[emotion_name] - neutral) * intensity

    return delta


def build_slider_emotion_vector(anchors, intensities):
    return anchors["Neutral"] + build_slider_delta(anchors, intensities)


def has_active_slider(intensities):
    return any(value > 0.0 for value in intensities.values())


def resolve_emotion_source(emotion_source, intensities):
    if emotion_source == "auto":
        return "audio-plus-sliders" if has_active_slider(intensities) else "audio"
    return emotion_source


def print_emotion_summary(label, emotion_vector, intensities=None):
    values = emotion_vector.detach().cpu().reshape(-1).tolist()
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    l2 = math.sqrt(sum(value * value for value in values))

    print(f"Emotion source: {label}")
    if intensities:
        active = {name: value for name, value in intensities.items() if value > 0.0}
        print(f"Active sliders: {active if active else 'none'}")
    print(
        "Vector stats: "
        f"min={min(values):.4f}, "
        f"max={max(values):.4f}, "
        f"mean={sum(values) / len(values):.4f}, "
        f"rms={rms:.4f}, "
        f"l2={l2:.4f}"
    )
    print("first 8:", [round(value, 4) for value in values[:8]])


def is_finger_joint(joint_name):
    return "Hand" in joint_name and joint_name not in {"RightHand", "LeftHand"}


def anchor_generated_channels(stylized_motion_real, original_motion, joint_names, anchor_root, anchor_fingers):
    if anchor_root:
        stylized_motion_real[:, :, :, 0] = original_motion[:, :, :, 0]

    if anchor_fingers:
        for joint_index, joint_name in enumerate(joint_names):
            if is_finger_joint(joint_name):
                stylized_motion_real[:, :, :, joint_index] = original_motion[:, :, :, joint_index]


def load_motion_models(device, num_joints, content_encoder_weights, generator_weights):
    print("Loading motion networks...")

    content_encoder = ContentEncoder(num_joints=num_joints).to(device)
    content_encoder.load_state_dict(torch.load(content_encoder_weights, map_location=device))
    content_encoder.eval()

    generator = MotionGenerator(num_joints=num_joints).to(device)
    generator.load_state_dict(torch.load(generator_weights, map_location=device))
    generator.eval()

    return content_encoder, generator


def choose_emotion_vector(
    emotion_source,
    audio_path,
    anchors_path,
    ast_weights,
    intensities,
    device,
):
    anchors = None
    slider_delta = None
    slider_vector = None
    if emotion_source in {"sliders", "audio-plus-sliders"}:
        anchors = load_emotion_anchors(anchors_path, device)
        slider_delta = build_slider_delta(anchors, intensities)
        slider_vector = anchors["Neutral"] + slider_delta

    audio_vector = None
    if emotion_source in {"audio", "audio-plus-sliders"}:
        audio_vector = load_audio_emotion_vector(audio_path, ast_weights, device)

    if emotion_source == "audio":
        emotion_vector = audio_vector
    elif emotion_source == "sliders":
        emotion_vector = slider_vector
    elif emotion_source == "audio-plus-sliders":
        emotion_vector = audio_vector + slider_delta
    else:
        raise ValueError(f"Unknown emotion source: {emotion_source}")

    print_emotion_summary(emotion_source, emotion_vector, intensities)
    return emotion_vector


def run_inference(
    audio_path,
    neutral_bvh_path,
    output_path,
    anchors_path,
    emotion_source,
    emotion_intensities,
    style_multiplier,
    fps,
    anchor_root,
    anchor_fingers,
    motion_stats_path,
    ast_weights,
    content_encoder_weights,
    generator_weights,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    neutral_bvh_path = resolve_existing_path(neutral_bvh_path, "Neutral BVH")
    output_path = resolve_output_path(output_path)
    motion_stats_path = resolve_existing_path(motion_stats_path, "Motion stats")
    content_encoder_weights = resolve_existing_path(content_encoder_weights, "Content encoder weights")
    generator_weights = resolve_existing_path(generator_weights, "Motion generator weights")

    resolved_emotion_source = resolve_emotion_source(emotion_source, emotion_intensities)

    if resolved_emotion_source in {"sliders", "audio-plus-sliders"}:
        anchors_path = resolve_existing_path(anchors_path, "Emotion anchors")

    if resolved_emotion_source in {"audio", "audio-plus-sliders"}:
        audio_path = resolve_existing_path(audio_path, "Audio file")
        ast_weights = resolve_existing_path(ast_weights, "AST emotion extractor weights")

    print("Loading BVH data and normalization statistics...")
    parser = BVHMotionParser(str(neutral_bvh_path))
    adj_matrix = parser.get_adjacency_matrix().to(device)
    num_joints = len(parser.joints)
    original_motion = parser.get_motion_tensor().unsqueeze(0).to(device)

    stats = torch.load(motion_stats_path, map_location=device)
    data_mean = stats["mean"].to(device)
    data_std = stats["std"].to(device)

    content_encoder, generator = load_motion_models(
        device,
        num_joints,
        content_encoder_weights,
        generator_weights,
    )

    emotion_vector = choose_emotion_vector(
        resolved_emotion_source,
        audio_path,
        anchors_path,
        ast_weights,
        emotion_intensities,
        device,
    )

    print("Generating stylized motion...")
    with torch.no_grad():
        original_motion_norm = (original_motion - data_mean) / data_std
        content_code = content_encoder(original_motion_norm, adj_matrix)
        stylized_motion_norm = generator(
            content_code,
            emotion_vector,
            adj_matrix,
            style_multiplier=style_multiplier,
        )
        stylized_motion_real = (stylized_motion_norm * data_std) + data_mean

        anchor_generated_channels(
            stylized_motion_real,
            original_motion,
            parser.joints,
            anchor_root=anchor_root,
            anchor_fingers=anchor_fingers,
        )

    print("Exporting to BVH format...")
    export_to_bvh(str(neutral_bvh_path), str(output_path), stylized_motion_real.squeeze(0), fps=fps)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run emo-anim motion inference from audio, terminal emotion sliders, "
            "or audio plus terminal slider offsets."
        )
    )

    parser.add_argument("--audio", default=DEFAULT_AUDIO_FILE, help="Input wav file for audio emotion extraction.")
    parser.add_argument("--neutral-bvh", default=DEFAULT_NEUTRAL_BVH, help="Input neutral/content BVH.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output BVH path.")
    parser.add_argument(
        "--anchors",
        default=str(DEFAULT_ANCHORS_FILE),
        help="Emotion anchor JSON path. Prefer master_emotion_anchors_raw.json for motion inference.",
    )
    parser.add_argument(
        "--emotion-source",
        choices=("auto", "audio", "sliders", "audio-plus-sliders"),
        default="auto",
        help=(
            "auto keeps old audio behavior unless any slider is nonzero. "
            "When sliders are nonzero, auto adds slider deltas to the extracted audio vector. "
            "sliders uses Neutral + slider deltas without audio."
        ),
    )
    parser.add_argument("--style-multiplier", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--motion-stats", default=str(SCRIPT_DIR / "motion_stats.pt"))
    parser.add_argument("--ast-weights", default=str(SCRIPT_DIR / "ast_emotion_extractor_weights.pth"))
    parser.add_argument("--content-weights", default=str(SCRIPT_DIR / "content_encoder_epoch_20.pth"))
    parser.add_argument("--generator-weights", default=str(SCRIPT_DIR / "motion_generator_epoch_20.pth"))
    parser.add_argument("--no-root-anchor", action="store_true")
    parser.add_argument("--no-finger-anchor", action="store_true")

    for emotion_name in EMOTION_NAMES:
        parser.add_argument(
            f"--{emotion_name.lower()}",
            type=zero_to_one,
            default=0.0,
            help=f"{emotion_name} slider intensity from 0 to 1.",
        )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    intensities = {name: getattr(args, name.lower()) for name in EMOTION_NAMES}

    run_inference(
        audio_path=args.audio,
        neutral_bvh_path=args.neutral_bvh,
        output_path=args.output,
        anchors_path=args.anchors,
        emotion_source=args.emotion_source,
        emotion_intensities=intensities,
        style_multiplier=args.style_multiplier,
        fps=args.fps,
        anchor_root=not args.no_root_anchor,
        anchor_fingers=not args.no_finger_anchor,
        motion_stats_path=args.motion_stats,
        ast_weights=args.ast_weights,
        content_encoder_weights=args.content_weights,
        generator_weights=args.generator_weights,
    )
