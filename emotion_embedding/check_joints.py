from bvh_loader import BVHMotionParser

# Point this to any valid .bvh file in your raw dataset
sample_bvh = r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_1_1.bvh"

print(f"Parsing BVH: {sample_bvh}...")
parser = BVHMotionParser(sample_bvh)

print(f"\nExtraction Complete! Found {len(parser.joints)} joints.")
print("=" * 40)
print("JOINT INDEX MAP")
print("=" * 40)

# enumerate() automatically pairs the index number with the joint name
for index, joint_name in enumerate(parser.joints):
    # Formats the output so the numbers align cleanly in your terminal
    print(f"Index {index:02d}  ->  {joint_name}")