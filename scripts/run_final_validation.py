from ultralytics import YOLO
import os

def main():
    print("=====================================================================")
    print("             YOLOv11 SUPER DATASET MODEL FINAL VALIDATION            ")
    print("=====================================================================")
    
    model_path = r"C:\FYP_v2\runs\detect\yolov11_super_dataset\weights\best.pt"
    data_config = r"C:\FYP_v2\external_datasets\super_yolo_dataset\data.yaml"
    project_dir = r"C:\FYP_v2\runs\detect\yolov11_super_dataset_val"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at: {model_path}")
        return
        
    print(f"[INFO] Loading best trained model weights from: {model_path}...")
    model = YOLO(model_path)
    
    print(f"[INFO] Starting fast validation on: {data_config}...")
    print("[INFO] This will compute final metrics and generate all charts (PR Curve, Confusion Matrix, etc.)")
    
    # Run validation
    metrics = model.val(
        data=data_config,
        project=r"C:\FYP_v2\runs\detect",
        name="yolov11_super_dataset_validation_plots",
        exist_ok=True,
        device=0,
        plots=True # Force drawing all plots
    )
    
    print("\n=====================================================================")
    print("🟢 VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"🟢 All charts and diagrams have been saved in: C:\\FYP_v2\\runs\\detect\\yolov11_super_dataset_validation_plots")
    print("=====================================================================")

if __name__ == "__main__":
    main()
