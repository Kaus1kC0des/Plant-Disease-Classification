import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
import io
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from IPython.display import clear_output

class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseClassifier, self).__init__()
        # Load EfficientNetB0 model
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Modify classifier to match the number of classes
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Number of output classes
        )

    def forward(self, x):
        return self.base_model(x)

    # Method to plot training history
    def plot_history(self, history):
        clear_output(wait=True)
        plt.figure(figsize=(12, 6))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.show()

    # Method to calculate accuracy
    def accuracy_fn(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        correct = (preds == labels).float()
        accuracy = correct.sum() / len(correct)
        return accuracy

    # Training method
    def train_model(self, criterion, optimizer, train_loader, val_loader, epochs, device, patience=5, min_delta=0):
        self.to(device)
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

        for epoch in range(epochs):
            self.train()  # Set model to training mode
            train_loss = 0
            train_accuracy = 0
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=True)

            # Training loop
            for inputs, labels in train_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                train_preds = self(inputs)
                loss = criterion(train_preds, labels)
                train_loss += loss.item() * inputs.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_accuracy += self.accuracy_fn(train_preds, labels).item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            train_accuracy /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)

            # Validation loop
            self.eval()  # Set model to evaluation mode
            val_loss = 0
            val_accuracy = 0
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation", leave=True)

            with torch.no_grad():
                for inputs, labels in val_loader_tqdm:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_accuracy += self.accuracy_fn(outputs, labels).item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_accuracy /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Log validation metrics to W&B
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epochs": epoch + 1})
            wandb.log({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy, "epochs": epoch + 1})

            tqdm.write(
                f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

            time.sleep(5)  # To ensure tqdm prints properly
            self.plot_history(history)  # Plot the training and validation metrics after each epoch

        return history, self


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


num_classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]

model = PlantDiseaseClassifier(num_classes=len(num_classes))
model.load_state_dict(torch.load(
    './Models/effnet_l_weights.pth', map_location=torch.device('cpu')))
model.eval()


def predict_image(img):
    img_pil = Image.open(io.BytesIO(img))
    tensor = transform(img_pil)
    xb = tensor.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return num_classes[preds[0].item()]
