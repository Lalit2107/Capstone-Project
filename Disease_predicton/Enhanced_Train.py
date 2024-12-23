import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the data transformations for enhanced dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit the pre-trained models like ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for pre-trained models
])

# Load the dataset (using your enhanced dataset)
train_data = datasets.ImageFolder(root=r"D:/Enhanced_Dataset_Split/train", transform=transform)
val_data = datasets.ImageFolder(root=r"D:/Enhanced_Dataset_Split/val", transform=transform)
test_data = datasets.ImageFolder(root=r"D:/Enhanced_Dataset_Split/test", transform=transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the CNN model (use a pre-trained model for better accuracy)
model = models.resnet18(pretrained=True)  # Use ResNet18 as an example
model.fc = nn.Linear(512, 4)  # Assuming 4 output classes: covid, normal, pneumonia, tb

model = model.to(device)  # Move the model to GPU if available

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm to display the progress bar
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear the gradients
        
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        # Accumulate loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar description with current loss and accuracy
        progress_bar.set_postfix({'loss': running_loss / (total / batch_size), 'accuracy': 100. * correct / total})
    
    return running_loss / len(train_loader), 100. * correct / total

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", unit="batch")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({'loss': running_loss / (total / batch_size), 'accuracy': 100. * correct / total})
    
    return running_loss / len(val_loader), 100. * correct / total

# Function to test the model on the test dataset
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({'accuracy': 100. * correct / total})
    
    return 100. * correct / total

# Main training loop
epochs = 10
best_val_acc = 0.0

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Train the model
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    
    # Validate the model
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    # Save the model if it achieves the best validation accuracy
    if val_acc > best_val_acc:
        print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%, saving model.")
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Test the model after training
test_acc = test(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.2f}%")
