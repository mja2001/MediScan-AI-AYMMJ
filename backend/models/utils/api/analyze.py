from fastapi import APIRouter, UploadFile, File
from models.chest_xray import predict as chest_predict
# Import others similarly
from utils.explain import generate_heatmap
from boto3 import client as boto_client
import os

router = APIRouter()

s3 = boto_client('s3')

@router.post("/")
async def analyze_image(file: UploadFile = File(...), modality: str = 'chest_xray'):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    if modality == 'chest_xray':
        preds = chest_predict(file_path)
    # Elif for other models
    
    heatmap = generate_heatmap(model, input_tensor, target_class=preds.argmax())
    
    # Upload to S3
    s3.upload_file(file_path, 'mediscan-bucket', file.filename)
    
    os.remove(file_path)
    return {"predictions": preds.tolist(), "confidence": confidence_score(preds), "heatmap": heatmap}
