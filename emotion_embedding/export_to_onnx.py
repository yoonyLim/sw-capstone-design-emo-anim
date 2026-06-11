import torch
import torch.nn as nn
from bvh_loader import BVHMotionParser
from content_encoder import ContentEncoder
from motion_generator import MotionGenerator

# ==========================================
# 1. THE C++ WRAPPER MODULE
# ==========================================
class UE5StylizationWrapper(nn.Module):
    def __init__(self, content_encoder, generator, adj_matrix):
        super().__init__()
        self.content_encoder = content_encoder
        self.generator = generator
        
        # Bake the matrix directly into the ONNX file
        self.register_buffer("adj_matrix", adj_matrix)

    def forward(self, neutral_motion, emotion_vector, style_multiplier):
        """
        Input 1: neutral_motion   [1, 3, 128, 75]
        Input 2: emotion_vector   [1, 64]
        Input 3: style_multiplier [1]
        """
        # 1. Strip the style
        content_code = self.content_encoder(neutral_motion, self.adj_matrix)
        
        # 2. Inject the amplified emotion
        # FIX: We strictly pass content_code, emotion_vector, adj_matrix, and the multiplier.
        # original_motion is completely removed from the generator call.
        stylized_motion = self.generator(content_code, emotion_vector, self.adj_matrix, style_multiplier)
        
        return stylized_motion

# ==========================================
# 2. THE EXPORT FUNCTION
# ==========================================
def export_to_unreal():
    device = torch.device("cpu") # CPU export for universal ONNX compatibility
    print("Preparing models for Unreal Engine 5 ONNX Export...")

    # 1. Grab the Adjacency Matrix
    sample_bvh = r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_1_1.bvh"
    parser = BVHMotionParser(sample_bvh)
    adj_matrix = parser.get_adjacency_matrix().to(device)
    num_joints = len(parser.joints)

    # 2. Load Your Trained Weights
    content_encoder = ContentEncoder(num_joints=num_joints).to(device)
    content_encoder.load_state_dict(torch.load("content_encoder_weights.pth", map_location=device))
    content_encoder.eval()

    generator = MotionGenerator(num_joints=num_joints).to(device)
    generator.load_state_dict(torch.load("motion_generator_weights.pth", map_location=device))
    generator.eval()

    # 3. Wrap them together
    ue5_model = UE5StylizationWrapper(content_encoder, generator, adj_matrix).to(device)
    ue5_model.eval()

    # 4. Create Dummy Inputs for the Tracer
    dummy_motion = torch.randn(1, 3, 128, num_joints).to(device) 
    dummy_emotion = torch.randn(1, 64).to(device)
    dummy_multiplier = torch.tensor([1.0], dtype=torch.float32).to(device)

    # 5. Export to ONNX
    export_path = "MotionStylizer_UE5.onnx"
    print(f"Tracing mathematical graphs and compiling to {export_path}...")
    
    torch.onnx.export(
        ue5_model,                                       
        (dummy_motion, dummy_emotion, dummy_multiplier), 
        export_path,                                     
        export_params=True,                              
        opset_version=17,                                
        do_constant_folding=True,                        
        input_names=['NeutralMotion', 'EmotionVector', 'StyleMultiplier'],  
        output_names=['StylizedMotion'],
        dynamic_axes={                                   
            'NeutralMotion': {2: 'time'},
            'StylizedMotion': {2: 'time'}
        }
    )
    
    print("\nSUCCESS: ONNX Export Complete!")

if __name__ == "__main__":
    export_to_unreal()