from fastapi import FastAPI, UploadFile
import torch
from core.model import SelfPruningCNN
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

model = SelfPruningCNN()
model.load_state_dict(torch.load("outputs/model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()

    return {"prediction": classes[pred]}
