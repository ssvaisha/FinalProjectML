import torch
import torch.nn.functional as F
from PIL import Image

from model import ConvModel, transform_eval

# same classes as your dataset
classes = ["COVID19","NORMAL","PNEUMONIA","TUBERCULOSIS"]

# check device
if torch.cuda.is_available():
    device = "cuda"
    print("cuda is available. using gpu.")
else:
    device = "cpu"
    print("cuda not available. using cpu.")

# load model
model = ConvModel().to(device)

# load trained weights
model.load_state_dict(torch.load("model.pth", map_location=device))

# evaluation mode
model.eval()

# open image
image_path = "sample_image.png"
image = Image.open(image_path).convert("RGB")

# apply same eval transform
image_tensor = transform_eval(image)

# add batch dimension
image_tensor = image_tensor.unsqueeze(0).to(device)

with torch.no_grad():

    outputs = model(image_tensor)

    # convert outputs to probabilities
    probs = F.softmax(outputs, dim=1)

    probs = probs[0].cpu()

# print prediction
for i, class_name in enumerate(classes):
    percent = probs[i].item() * 100
    print(f"{class_name}: {percent:.2f}%")

# predicted class
predicted = torch.argmax(probs).item()

print("\nPrediction:", classes[predicted])