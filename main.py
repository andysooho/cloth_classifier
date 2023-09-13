from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
from classifier import classify_image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
async def hello():
    return JSONResponse(content={"message": "Hello World!"})


@app.post("/classify")
async def upload_image(image: UploadFile = File(...)):
    image_path = Path("uploaded_images") / image.filename
    with image_path.open("wb") as f:
        f.write(await image.read())

    predicted_class_idx, predicted_label = classify_image(image_path)

    return JSONResponse(content={"predicted_class_index": predicted_class_idx, "predicted_label": predicted_label})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4282)
