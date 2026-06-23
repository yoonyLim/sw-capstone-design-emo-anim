from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

from bvh_loader import BVHMotionParser


sample_bvh = str(get_path("EMO_ANIM_SAMPLE_BVH", "data/sample.bvh"))

print(f"Parsing BVH: {sample_bvh}...")
parser = BVHMotionParser(sample_bvh)

print(f"\nExtraction Complete! Found {len(parser.joints)} joints.")
print("=" * 40)
print("JOINT INDEX MAP")
print("=" * 40)


for index, joint_name in enumerate(parser.joints):

    print(f"Index {index:02d}  ->  {joint_name}")
