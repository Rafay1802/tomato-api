from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import requests

app = FastAPI()

# Allow CORS (for frontend connection like WordPress, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now. You can restrict later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL of your uploaded model on Dropbox
MODEL_URL = "https://www.dropbox.com/scl/fi/v0b2ja6vzvbsfzx27tuw1/60epochTomatoFruitResnet101v2.keras?rlkey=vaqtvjfk0x2ob847ent9x67in&st=95mllvpj&dl=1"

# Local path to save the model
MODEL_PATH = "model.keras"

# Download the model from Dropbox if not already downloaded
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Dropbox...")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print(f"Failed to download model. Status code: {response.status_code}")
        raise Exception("Could not download model from Dropbox.")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded into memory.")

# Define your class labels
labels = ['Ripe', 'Unripe', 'Old', 'Damaged']  # ✏️ Make sure these are your real classes

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize the pixel values
        input_tensor = np.expand_dims(img_array, axis=0)  # Add batch dimension

        prediction = model.predict(input_tensor)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "class": labels[class_id],
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
