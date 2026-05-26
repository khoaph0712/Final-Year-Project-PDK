import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# Setup paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))

# Load local models & paths
ENSEMBLE_DIR = ROOT_DIR / "runs" / "dl" / "convnext_ensemble"
OUTPUT_ONNX_PATH = ENSEMBLE_DIR / "unified_multimodal_waste_pipeline.onnx"

class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtFeatureExtractor, self).__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.backbone(x)
        return torch.flatten(features, 1)

class UnifiedMultimodalWastePipeline(nn.Module):
    """
    Unified Multi-Modal Computational Graph exporting both Spatial (ConvNeXt) 
    and Material (Handcrafted) inputs as a single, compiled edge-ready model.
    """
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(UnifiedMultimodalWastePipeline, self).__init__()
        self.convnext_extractor = ConvNeXtFeatureExtractor()
        self.input_dim = 768 + 637
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, image_tensor, handcrafted_features_tensor):
        # 1. Spatial Mode Forward Pass
        deep_features = self.convnext_extractor(image_tensor)
        
        # 2. Parallel Concatenation (Feature-Level Fusion)
        fused_super_vector = torch.cat((deep_features, handcrafted_features_tensor), dim=1)
        
        # 3. Fused Inference Head
        logits = self.classifier(fused_super_vector)
        return logits

def main():
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("Unified Multi-Modal Pipeline: ONNX Graph Exporter")
    print("====================================================")
    
    # 1. Initialize and load trained state weights
    print("[INFO] Constructing PyTorch Multi-Modal Model...")
    model = UnifiedMultimodalWastePipeline(num_classes=7)
    
    pytorch_weights = ENSEMBLE_DIR / "best_convnext_ensemble.pth"
    if pytorch_weights.exists():
        print(f"[INFO] Loading trained weights from: {pytorch_weights}")
        model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
    else:
        print("[WARNING] Trained weights not found. Exporting base graph with random initialization.")
        
    model.eval()

    # 2. Define dummy parallel inputs matching edge inputs
    print("[INFO] Defining parallel dummy inputs...")
    dummy_image = torch.randn(1, 3, 224, 224)              # Mode 1: Spatial crop
    dummy_handcrafted = torch.randn(1, 637)               # Mode 2: Material texture vectors
    
    # 3. Export to a single ONNX file
    print(f"[INFO] Compiling computational graph to ONNX at: {OUTPUT_ONNX_PATH}...")
    torch.onnx.export(
        model,
        (dummy_image, dummy_handcrafted),
        str(OUTPUT_ONNX_PATH),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['spatial_image_input', 'material_texture_input'],
        output_names=['class_logits_output'],
        dynamic_axes={
            'spatial_image_input': {0: 'batch_size'},
            'material_texture_input': {0: 'batch_size'},
            'class_logits_output': {0: 'batch_size'}
        }
    )
    
    print(f"\n[SUCCESS] Unified Fused Multi-Modal ONNX Model successfully saved!")
    print(f"  --> File Path: {OUTPUT_ONNX_PATH}")
    print("  --> Ready for Flutter / ONNX Runtime edge deployment (No switching lag)!")
    print("====================================================")

if __name__ == "__main__":
    main()
