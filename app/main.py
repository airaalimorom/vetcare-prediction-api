from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import json
import os
import random
import requests

app = FastAPI(title="Pet Disease Classifier API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_mapping = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def download_model_files():
    """Download model files from Google Drive if they don't exist"""
    print("🔄 Starting model file download...")
    os.makedirs('model', exist_ok=True)
    
    # Your Google Drive direct download links
    model_files = {
        'model/proper_medical_model.pth': 'https://drive.google.com/uc?export=download&id=1UZRn58UHXKFZ38661xNDW1WU-QrHWa3b',
        'model/proper_class_mapping.json': 'https://drive.google.com/uc?export=download&id=1PtWQq2Wk8IanKil6hsD_7RCG8IFnDcIe',
        'model/real_pet_disease_model.pth': 'https://drive.google.com/uc?export=download&id=1p2_wSpeNoftlByCLDcxdfk3nOG8rw9pN',
        'model/real_class_mapping.json': 'https://drive.google.com/uc?export=download&id=1dw46v0t6sIbjVMAkCzuBIGDQL-KiWq-G'
    }
    
    for file_path, url in model_files.items():
        print(f"📥 Checking {file_path}...")
        if not os.path.exists(file_path):
            print(f"   Downloading from {url}...")
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ Downloaded {file_path} ({file_size} bytes)")
                else:
                    print(f"   ❌ Failed to download {file_path}: HTTP {response.status_code}")
            except Exception as e:
                print(f"   ❌ Error downloading {file_path}: {e}")
        else:
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} already exists ({file_size} bytes)")
    
    # List all files in model directory
    print("📁 Files in model directory:")
    for file in os.listdir('model'):
        file_path = os.path.join('model', file)
        size = os.path.getsize(file_path)
        print(f"   - {file}: {size} bytes")

def load_model():
    """Load the trained model"""
    global model, class_mapping
    
    print("🔄 Starting model loading process...")
    
    try:
        # Try different model paths in order
        model_paths = [
            'model/proper_medical_model.pth',
            'model/real_pet_disease_model.pth', 
            'model/pet_disease_model.pth'
        ]
        
        class_mapping_paths = [
            'model/proper_class_mapping.json',
            'model/real_class_mapping.json',
            'model/class_mapping.json'
        ]
        
        model_path = None
        class_mapping_path = None
        
        # Find the first model that exists
        for i, path in enumerate(model_paths):
            if os.path.exists(path):
                model_path = path
                class_mapping_path = class_mapping_paths[i]
                print(f"✅ Found model: {model_path}")
                print(f"✅ Found class mapping: {class_mapping_path}")
                break
        
        if not model_path:
            print("❌ No model file found at any path!")
            print("📁 Current directory contents:")
            for item in os.listdir('.'):
                print(f"   - {item}")
            print("📁 Model directory contents:")
            if os.path.exists('model'):
                for item in os.listdir('model'):
                    print(f"   - {item}")
            else:
                print("   Model directory doesn't exist!")
            return False
        
        # Check if model files exist
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Load class mapping
        print("📖 Loading class mapping...")
        if not os.path.exists(class_mapping_path):
            print(f"❌ Class mapping file not found: {class_mapping_path}")
            return False
            
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        # Get number of classes
        num_classes = len(class_mapping['idx_to_label'])
        print(f"📊 Number of classes: {num_classes}")
        print(f"🏥 Classes: {list(class_mapping['label_to_idx'].keys())}")
        
        # Create model - Use EfficientNet for proper medical model
        print("🔨 Creating model architecture...")
        if 'proper' in model_path:
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            print("🔬 Using EfficientNet (medical optimized)")
        else:
            # Fallback to ResNet18 for other models
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            print("🔧 Using ResNet18 (compatible)")
        
        # Load trained weights
        print(f"📂 Loading model weights from: {model_path}")
        print(f"📂 File size: {os.path.getsize(model_path)} bytes")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Test the model with a dummy input
        print("🧪 Testing model with dummy input...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            dummy_output = model(dummy_input)
        print(f"🧪 Model test - Output shape: {dummy_output.shape}")
        print(f"🧪 Model test - Output range: {dummy_output.min().item():.3f} to {dummy_output.max().item():.3f}")
        
        print("✅ Model loaded successfully!")
        
        # Determine which model is being used
        if 'proper' in model_path:
            model_type = "PROPER MEDICAL"
        elif 'real' in model_path:
            model_type = "REAL" 
        else:
            model_type = "DEMO"
            
        print(f"💾 Using: {model_type} model")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        print(f"❌ Detailed error: {traceback.format_exc()}")
        print("⚠️  Running in demo mode")
        return False

def get_demo_prediction(filename):
    """Generate demo predictions when model isn't loaded"""
    print("🎭 Using demo mode for prediction")
    # ... keep your existing demo function code ...
    classes = [
        'Dental Disease in Cat', 'Dental Disease in Dog', 'distemper', 
        'Distemper in Dog', 'Ear Mites in Cat', 'ear_infection', 
        'Eye Infection in Cat', 'Eye Infection in Dog', 'Feline Leukemia',
        'Feline Panleukopenia', 'Fungal Infection in Cat', 'Fungal Infection in Dog',
        'healthy', 'Hot Spots in Dog', 'Kennel Cough in Dog', 'kennel_cough',
        'Mange in Dog', 'parvovirus', 'Parvovirus in Dog', 'Ringworm in Cat',
        'Scabies in Cat', 'Skin Allergy in Cat', 'Skin Allergy in Dog',
        'Tick Infestation in Dog', 'Urinary Tract Infection in Cat',
        'Worm Infection in Cat', 'Worm Infection in Dog'
    ]
    
    # Generate consistent "predictions" based on filename
    file_hash = hash(filename) % 100
    
    if file_hash < 15:
        primary_class = "Ear Mites in Cat"
        confidence = random.uniform(75, 90)
    elif file_hash < 30:
        primary_class = "Parvovirus in Dog"
        confidence = random.uniform(70, 85)
    elif file_hash < 45:
        primary_class = "Skin Allergy in Dog"
        confidence = random.uniform(65, 80)
    elif file_hash < 60:
        primary_class = "Dental Disease in Cat"
        confidence = random.uniform(60, 75)
    elif file_hash < 75:
        primary_class = "Kennel Cough in Dog"
        confidence = random.uniform(55, 70)
    else:
        primary_class = "healthy"
        confidence = random.uniform(80, 95)
    
    # Create predictions list
    predictions = []
    primary_idx = classes.index(primary_class)
    
    for i, cls in enumerate(classes):
        if cls == primary_class:
            pred_confidence = confidence
        else:
            pred_confidence = random.uniform(1, 20)  # Lower confidence for other classes
        
        predictions.append({
            "class": cls,
            "confidence": round(pred_confidence, 2),
            "class_id": i
        })
    
    # Sort by confidence (descending) and take top 5
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    predictions = predictions[:5]
    
    return predictions, predictions[0]

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    print("🚀 Starting Pet Disease Classifier API...")
    print("📥 Downloading model files...")
    download_model_files()
    print("🔧 Loading model...")
    load_model()
    print("✅ Startup complete!")

@app.get("/")
def root():
    return {
        "message": "Pet Disease Classifier API", 
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    model_type = "None"
    if model is not None:
        if 'efficientnet' in str(model.__class__).lower():
            model_type = "PROPER MEDICAL"
        elif 'resnet' in str(model.__class__).lower():
            model_type = "REAL"
        else:
            model_type = "DEMO"
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": str(device),
        "classes_available": len(class_mapping['label_to_idx']) if class_mapping else 0
    }

@app.get("/model-info")
def model_info():
    """Check model information"""
    if model is None:
        return {
            "error": "No model loaded",
            "debug_info": {
                "model_variable": str(model),
                "class_mapping_variable": str(class_mapping)
            }
        }
    
    model_info = {
        "model_type": model.__class__.__name__,
        "model_architecture": "EfficientNet" if 'efficientnet' in str(model.__class__).lower() else "ResNet",
        "num_classes": len(class_mapping['idx_to_label']) if class_mapping else 0,
        "classes_loaded": list(class_mapping['label_to_idx'].keys()) if class_mapping else [],
        "pretrained_used": True
    }
    
    # Check model file info
    model_files = {}
    for file in ['model/proper_medical_model.pth', 'model/proper_class_mapping.json']:
        if os.path.exists(file):
            model_files[file] = {
                "exists": True,
                "size": os.path.getsize(file),
                "modified": os.path.getmtime(file)
            }
        else:
            model_files[file] = {"exists": False}
    
    return {
        "model_info": model_info,
        "files": model_files,
        "device": str(device)
    }

@app.get("/classes")
def get_classes():
    if class_mapping:
        return {"classes": list(class_mapping['label_to_idx'].keys())}
    else:
        # Return the real classes from your dataset
        return {"classes": [
            'Dental Disease in Cat', 'Dental Disease in Dog', 'distemper', 
            'Distemper in Dog', 'Ear Mites in Cat', 'ear_infection', 
            'Eye Infection in Cat', 'Eye Infection in Dog', 'Feline Leukemia',
            'Feline Panleukopenia', 'Fungal Infection in Cat', 'Fungal Infection in Dog',
            'healthy', 'Hot Spots in Dog', 'Kennel Cough in Dog', 'kennel_cough',
            'Mange in Dog', 'parvovirus', 'Parvovirus in Dog', 'Ringworm in Cat',
            'Scabies in Cat', 'Skin Allergy in Cat', 'Skin Allergy in Dog',
            'Tick Infestation in Dog', 'Urinary Tract Infection in Cat',
            'Worm Infection in Cat', 'Worm Infection in Dog'
        ]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")
    
    try:
        # If model is not loaded, use demo mode
        if model is None:
            print("🎭 Using demo mode for prediction")
            predictions, primary_prediction = get_demo_prediction(file.filename)
            
            return {
                "success": True,
                "predictions": predictions,
                "primary_prediction": primary_prediction,
                "file_name": file.filename,
                "file_type": file.content_type,
                "message": "Demo mode - using sample predictions",
                "is_demo": True
            }
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make real prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(class_mapping['idx_to_label'])))
        
        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            class_name = class_mapping['idx_to_label'][str(idx.item())]
            predictions.append({
                "class": class_name,
                "confidence": round(prob.item() * 100, 2),
                "class_id": int(idx.item())
            })
        
        # Determine message based on model type
        model_type = "PROPER MEDICAL" if 'efficientnet' in str(model.__class__).lower() else "REAL"
        message = f"{model_type} model prediction - trained on medical images"
        
        # Add confidence warning for low confidence predictions
        if predictions[0]['confidence'] < 70:
            message += " ⚠️ Low confidence - consult veterinarian"
        
        print(f"🔍 Prediction result: {predictions[0]['class']} ({predictions[0]['confidence']}%)")
        
        return {
            "success": True,
            "predictions": predictions,
            "primary_prediction": predictions[0],
            "file_name": file.filename,
            "file_type": file.content_type,
            "message": message,
            "is_demo": False
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict multiple images at once"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    for file in files:
        try:
            # Use the predict function for each file
            result = await predict(file)
            results.append({
                "file_name": file.filename,
                "success": True,
                "prediction": result["primary_prediction"],
                "is_demo": result.get("is_demo", False)
            })
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
