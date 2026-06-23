from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torchaudio


THIS_FILE = Path(__file__).resolve()
WORKSPACE_DIR = THIS_FILE.parents[1]
DEFAULT_EMO_ANIM_DIR = WORKSPACE_DIR.parent / "emo-anim"
DEFAULT_EMOTION_EMBEDDING_DIR = DEFAULT_EMO_ANIM_DIR / "emotion_embedding"
DEFAULT_AUDIO_EMOTION_DIR = DEFAULT_EMO_ANIM_DIR / "audio_emotion_vector_extraction"


@dataclass
class EvalRow:
    bvh_path: str
    audio_path: str
    start_frame: int
    end_frame: int
    frames: int
    angular_mae_deg: float
    angular_rmse_deg: float
    angular_p95_deg: float
    velocity_mae_deg_per_frame: float
    velocity_p95_deg_per_frame: float
    content_l1: float


def add_import_paths(emotion_embedding_dir: Path, audio_emotion_dir: Path) -> None:
    sys.path.insert(0, str(emotion_embedding_dir))
    sys.path.insert(0, str(audio_emotion_dir))


def resolve_path(path_value: str | Path, description: str, must_exist: bool = True) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def find_audio_for_bvh(bvh_path: Path, bvh_dir: Path, audio_dir: Path) -> Path | None:
    wav_name = bvh_path.with_suffix(".wav").name

    try:
        rel_parent = bvh_path.parent.relative_to(bvh_dir)
    except ValueError:
        rel_parent = Path()

    candidates = [
        audio_dir / rel_parent / wav_name,
        audio_dir / bvh_path.parent.name / wav_name,
        audio_dir / wav_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def iter_pairs(bvh_dir: Path, audio_dir: Path, max_files: int | None) -> Iterable[tuple[Path, Path]]:
    count = 0
    for bvh_path in sorted(bvh_dir.rglob("*.bvh")):
        audio_path = find_audio_for_bvh(bvh_path, bvh_dir, audio_dir)
        if audio_path is None:
            continue

        yield bvh_path, audio_path
        count += 1
        if max_files is not None and count >= max_files:
            break


def iter_random_unpaired_pairs(
    bvh_dir: Path,
    audio_dir: Path,
    sample_files: int,
    seed: int,
) -> list[tuple[Path, Path]]:
    rng = random.Random(seed)
    bvh_paths = sorted(bvh_dir.rglob("*.bvh"))
    audio_paths = sorted(audio_dir.rglob("*.wav"))

    if not bvh_paths:
        raise RuntimeError(f"No BVH files found under: {bvh_dir}")
    if not audio_paths:
        raise RuntimeError(f"No WAV files found under: {audio_dir}")

    if sample_files > len(bvh_paths):
        print(f"Requested {sample_files} BVHs, but only {len(bvh_paths)} exist. Using all BVHs.")
        selected_bvhs = bvh_paths
    else:
        selected_bvhs = rng.sample(bvh_paths, sample_files)

    return [(bvh_path, rng.choice(audio_paths)) for bvh_path in selected_bvhs]


def load_audio_segment(
    audio_path: Path,
    start_frame: int,
    frame_count: int,
    fps: float,
    target_sr: int,
    device: torch.device,
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)

    start_sample = int((start_frame / fps) * target_sr)
    sample_count = int((frame_count / fps) * target_sr)
    segment = waveform[:, start_sample : start_sample + sample_count]

    if segment.shape[1] < sample_count:
        segment = torch.nn.functional.pad(segment, (0, sample_count - segment.shape[1]))

    return segment.unsqueeze(0).to(device)


def angular_error_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0
    return diff.abs()


def tensor_percentile(values: torch.Tensor, q: float) -> float:
    flat = values.detach().flatten()
    if flat.numel() == 0:
        return 0.0
    return torch.quantile(flat, q).item()


def evaluate_chunk(
    motion_real: torch.Tensor,
    audio_tensor: torch.Tensor,
    adj_matrix: torch.Tensor,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    audio_extractor: torch.nn.Module,
    content_encoder: torch.nn.Module,
    generator: torch.nn.Module,
    style_multiplier: float,
) -> tuple[torch.Tensor, float]:
    motion_norm = (motion_real - data_mean) / data_std

    with torch.no_grad():
        emotion_vector = audio_extractor(audio_tensor.view(audio_tensor.size(0), 1, -1))
        target_content = content_encoder(motion_norm, adj_matrix)
        output_norm = generator(target_content, emotion_vector, adj_matrix, style_multiplier=style_multiplier)
        generated_content = content_encoder(output_norm, adj_matrix)
        output_real = (output_norm * data_std) + data_mean
        content_l1 = torch.nn.functional.l1_loss(generated_content, target_content).item()

    return output_real, content_l1


def summarize(rows: list[EvalRow]) -> dict[str, float | int]:
    if not rows:
        return {"chunks": 0}

    def avg(name: str) -> float:
        return sum(getattr(row, name) for row in rows) / len(rows)

    return {
        "chunks": len(rows),
        "files": len({row.bvh_path for row in rows}),
        "angular_mae_deg": avg("angular_mae_deg"),
        "angular_rmse_deg": avg("angular_rmse_deg"),
        "angular_p95_deg": avg("angular_p95_deg"),
        "velocity_mae_deg_per_frame": avg("velocity_mae_deg_per_frame"),
        "velocity_p95_deg_per_frame": avg("velocity_p95_deg_per_frame"),
        "content_l1": avg("content_l1"),
    }


def write_outputs(rows: list[EvalRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "decoder_eval_chunks.csv"
    json_path = output_dir / "decoder_eval_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    summary = summarize(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved per-chunk CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def write_report_plots(rows: list[EvalRow], output_dir: Path) -> None:
    if not rows:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    def sorted_values(name: str) -> list[float]:
        return sorted(float(getattr(row, name)) for row in rows)

    metric_specs = [
        ("angular_mae_deg", "Angular MAE (degrees)", "decoder_eval_angular_mae.png"),
        ("angular_p95_deg", "Angular P95 Error (degrees)", "decoder_eval_angular_p95.png"),
        ("velocity_p95_deg_per_frame", "Velocity P95 Error (deg/frame)", "decoder_eval_velocity_p95.png"),
        ("content_l1", "Content Code L1 Error", "decoder_eval_content_l1.png"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Decoder-side Evaluation on Random BVH Samples", fontsize=18)
    for ax, (metric, title, _) in zip(axes.flatten(), metric_specs):
        values = sorted_values(metric)
        ax.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.5, markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Samples sorted by error")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    combined_path = output_dir / "decoder_eval_metric_curves.png"
    fig.savefig(combined_path, dpi=160)
    plt.close(fig)

    box_values = [sorted_values(metric) for metric, _, _ in metric_specs[:3]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(box_values, labels=["MAE", "P95", "Vel P95"], showmeans=True)
    ax.set_title("Decoder Error Distribution")
    ax.set_ylabel("Degrees / degree-per-frame")
    ax.grid(True, axis="y", alpha=0.3)
    box_path = output_dir / "decoder_eval_error_boxplot.png"
    fig.tight_layout()
    fig.savefig(box_path, dpi=160)
    plt.close(fig)

    audio_groups: dict[str, list[EvalRow]] = {}
    for row in rows:
        audio_groups.setdefault(Path(row.audio_path).stem, []).append(row)

    audio_labels = sorted(audio_groups)
    audio_mae = [
        sum(row.angular_mae_deg for row in audio_groups[label]) / len(audio_groups[label])
        for label in audio_labels
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(audio_labels, audio_mae, color="#5f91cc")
    ax.set_title("Mean Angular MAE by Test Audio")
    ax.set_ylabel("Angular MAE (degrees)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    audio_path = output_dir / "decoder_eval_by_audio.png"
    fig.tight_layout()
    fig.savefig(audio_path, dpi=160)
    plt.close(fig)

    table_path = output_dir / "decoder_eval_summary_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "median", "p95", "max"])
        for metric, title, _ in metric_specs:
            values = sorted_values(metric)
            n = len(values)
            median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
            p95 = values[min(n - 1, math.ceil(n * 0.95) - 1)]
            writer.writerow([title, sum(values) / n, median, p95, max(values)])

    print(f"Saved plots: {combined_path}, {box_path}, {audio_path}")
    print(f"Saved summary table: {table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate emo-anim decoder-side reconstruction on paired BVH/WAV train data.")
    parser.add_argument("--emo-anim-dir", default=str(DEFAULT_EMO_ANIM_DIR))
    parser.add_argument("--bvh-dir", required=True, help="Root folder containing BEAT BVH files.")
    parser.add_argument("--audio-dir", required=True, help="Root folder containing paired BEAT WAV files.")
    parser.add_argument("--output-dir", default=str(WORKSPACE_DIR / "ReportVid" / "decoder_eval"))
    parser.add_argument("--motion-stats", default=None)
    parser.add_argument("--ast-weights", default=None)
    parser.add_argument("--content-weights", default=None)
    parser.add_argument("--generator-weights", default=None)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching; current implementation runs one chunk at a time.")
    parser.add_argument("--style-multiplier", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--random-sample-files", type=int, default=None, help="Randomly sample this many BVH files and assign WAV files from --audio-dir.")
    parser.add_argument("--random-one-window-per-file", action="store_true", help="Evaluate one random window from each selected BVH instead of every stride window.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    emo_anim_dir = resolve_path(args.emo_anim_dir, "emo-anim directory")
    emotion_embedding_dir = emo_anim_dir / "emotion_embedding"
    audio_emotion_dir = emo_anim_dir / "audio_emotion_vector_extraction"
    add_import_paths(emotion_embedding_dir, audio_emotion_dir)

    from ast_model import ASTEmotionExtractor
    from bvh_loader import BVHMotionParser
    from content_encoder import ContentEncoder
    from motion_generator import MotionGenerator

    bvh_dir = resolve_path(args.bvh_dir, "BVH directory")
    audio_dir = resolve_path(args.audio_dir, "Audio directory")
    output_dir = resolve_path(args.output_dir, "Output directory", must_exist=False)

    motion_stats_path = resolve_path(args.motion_stats or emotion_embedding_dir / "motion_stats.pt", "Motion stats")
    ast_weights_path = resolve_path(args.ast_weights or emotion_embedding_dir / "ast_emotion_extractor_weights.pth", "AST weights")
    content_weights_path = resolve_path(args.content_weights or emotion_embedding_dir / "content_encoder_final.pth", "Content encoder weights")
    generator_weights_path = resolve_path(args.generator_weights or emotion_embedding_dir / "motion_generator_final.pth", "Motion generator weights")

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"BVH root: {bvh_dir}")
    print(f"Audio root: {audio_dir}")

    if args.random_sample_files is not None:
        pairs = iter_random_unpaired_pairs(
            bvh_dir,
            audio_dir,
            sample_files=args.random_sample_files,
            seed=args.seed,
        )
    else:
        pairs = list(iter_pairs(bvh_dir, audio_dir, args.max_files))

    if not pairs:
        raise RuntimeError("No paired BVH/WAV files were found.")

    template_parser = BVHMotionParser(str(pairs[0][0]))
    num_joints = len(template_parser.joints)
    adj_matrix = template_parser.get_adjacency_matrix().to(device)

    stats = torch.load(motion_stats_path, map_location=device)
    data_mean = stats["mean"].to(device)
    data_std = stats["std"].to(device)
    data_std = torch.where(data_std.abs() < 1e-6, torch.ones_like(data_std), data_std)

    audio_extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    audio_extractor.load_state_dict(torch.load(ast_weights_path, map_location=device))
    audio_extractor.eval()

    content_encoder = ContentEncoder(num_joints=num_joints).to(device)
    content_encoder.load_state_dict(torch.load(content_weights_path, map_location=device))
    content_encoder.eval()

    generator = MotionGenerator(num_joints=num_joints).to(device)
    generator.load_state_dict(torch.load(generator_weights_path, map_location=device))
    generator.eval()

    rng = random.Random(args.seed)
    rows: list[EvalRow] = []
    total_chunks = 0

    for file_index, (bvh_path, audio_path) in enumerate(pairs, start=1):
        print(f"\n[{file_index}/{len(pairs)}] {bvh_path.name}")
        parser = BVHMotionParser(str(bvh_path))
        motion = parser.get_motion_tensor().unsqueeze(0).to(device)
        total_frames = motion.shape[2]

        if motion.shape[3] != num_joints:
            print(f"  Skipping: expected {num_joints} joints, found {motion.shape[3]}")
            continue

        valid_starts = list(range(0, max(0, total_frames - args.window_size + 1), args.stride))
        if not valid_starts:
            print(f"  Skipping: only {total_frames} frames, need {args.window_size}.")
            continue

        if args.random_one_window_per_file:
            starts = [rng.choice(valid_starts)]
        else:
            starts = valid_starts

        for start_frame in starts:
            end_frame = start_frame + args.window_size
            chunk = motion[:, :, start_frame:end_frame, :]
            audio_chunk = load_audio_segment(
                audio_path,
                start_frame=start_frame,
                frame_count=args.window_size,
                fps=args.fps,
                target_sr=args.target_sr,
                device=device,
            )

            generated, content_l1 = evaluate_chunk(
                chunk,
                audio_chunk,
                adj_matrix,
                data_mean,
                data_std,
                audio_extractor,
                content_encoder,
                generator,
                style_multiplier=args.style_multiplier,
            )

            error = angular_error_deg(generated, chunk)
            velocity_error = angular_error_deg(
                generated[:, :, 1:, :] - generated[:, :, :-1, :],
                chunk[:, :, 1:, :] - chunk[:, :, :-1, :],
            )

            row = EvalRow(
                bvh_path=str(bvh_path),
                audio_path=str(audio_path),
                start_frame=start_frame,
                end_frame=end_frame,
                frames=args.window_size,
                angular_mae_deg=error.mean().item(),
                angular_rmse_deg=math.sqrt(torch.mean(error * error).item()),
                angular_p95_deg=tensor_percentile(error, 0.95),
                velocity_mae_deg_per_frame=velocity_error.mean().item(),
                velocity_p95_deg_per_frame=tensor_percentile(velocity_error, 0.95),
                content_l1=content_l1,
            )
            rows.append(row)
            total_chunks += 1

            if args.max_chunks is not None and total_chunks >= args.max_chunks:
                write_outputs(rows, output_dir)
                write_report_plots(rows, output_dir)
                return

    write_outputs(rows, output_dir)
    write_report_plots(rows, output_dir)


if __name__ == "__main__":
    main()
