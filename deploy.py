from typing import Annotated

from fastapi import FastAPI, File, UploadFile
import numpy
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import Image
from torchvision import transforms
from PIL import Image
import cv2
import os
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from torchvision.transforms import ToPILImage
from image_sr import enhance
from io import BytesIO



class UploadFileRequest(BaseModel):
    uniqueID: str
    file: UploadFile

app = FastAPI()
# Allow your frontend origin for development

app.mount("/images", StaticFiles(directory="results"), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    output="results"
    img_path=f"{output}/{file.filename}"

    os.makedirs(output, exist_ok=True)

    with open(img_path,'wb') as f:
        contents=file.file.read()
        f.write(contents)
    

    img=Image.open(img_path)

    enhance(img_path)      
 

    return FileResponse(img_path, media_type="image/jpeg")

@app.get("/")
def home():
    return "Hello"