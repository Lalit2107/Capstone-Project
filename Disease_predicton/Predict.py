import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import cv2  # For OpenCV functions

# Load the trained model
def load_model(model_path, device):
    model = models.resnet18(pretrained=False)  # Use the same architecture as training
    model.fc = nn.Linear(512, 4)  # Adjust based on the number of classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input image (resize, normalize, etc.)
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    
    # Apply Histogram Equalization
    equalized_image = cv2.equalizeHist(image)
    
    # Convert the equalized image to RGB (since ResNet expects 3 channels)
    equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

    # Define the transformation (resize, normalize, and convert to tensor)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Same normalization as training
    ])
    
    # Apply the transformation
    input_tensor = transform(equalized_image_rgb).unsqueeze(0)  # Add batch dimension
    return input_tensor

# Fuzzy logic system setup
def setup_fuzzy_system():
    # Define the fuzzy variables (input and output)
    prob_covid = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_covid')
    prob_normal = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_normal')
    prob_pneumonia = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_pneumonia')
    prob_tb = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_tb')

    diagnosis = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'diagnosis')

    # Define fuzzy membership functions for each variable
    prob_covid['low'] = fuzz.trimf(prob_covid.universe, [0, 0, 0.5])
    prob_covid['high'] = fuzz.trimf(prob_covid.universe, [0.5, 1, 1])
    
    prob_normal['low'] = fuzz.trimf(prob_normal.universe, [0, 0, 0.5])
    prob_normal['high'] = fuzz.trimf(prob_normal.universe, [0.5, 1, 1])
    
    prob_pneumonia['low'] = fuzz.trimf(prob_pneumonia.universe, [0, 0, 0.5])
    prob_pneumonia['high'] = fuzz.trimf(prob_pneumonia.universe, [0.5, 1, 1])
    
    prob_tb['low'] = fuzz.trimf(prob_tb.universe, [0, 0, 0.5])
    prob_tb['high'] = fuzz.trimf(prob_tb.universe, [0.5, 1, 1])
    
    diagnosis['normal'] = fuzz.trimf(diagnosis.universe, [0, 0, 0.33])
    diagnosis['mild'] = fuzz.trimf(diagnosis.universe, [0.33, 0.5, 0.67])
    diagnosis['severe'] = fuzz.trimf(diagnosis.universe, [0.67, 1, 1])

    # Define fuzzy rules
    rule1 = ctrl.Rule(prob_normal['high'], diagnosis['normal'])
    rule2 = ctrl.Rule(prob_pneumonia['high'], diagnosis['mild'])
    rule3 = ctrl.Rule(prob_covid['high'], diagnosis['severe'])
    rule4 = ctrl.Rule(prob_tb['high'], diagnosis['severe'])
    
    # Combine the rules
    diagnosis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    diagnosis_sim = ctrl.ControlSystemSimulation(diagnosis_ctrl)
    
    return diagnosis_sim

# Predict the disease using CNN and fuzzy logic
def predict_disease(image_path, model, device):
    # Preprocess the image (with histogram equalization)
    image = preprocess_image(image_path).to(device)
    
    # Get the CNN output (probabilities for each class)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # Convert to probabilities
    
    # Mapping the probabilities to their respective classes
    prob_covid, prob_normal, prob_pneumonia, prob_tb = probabilities
    
    # Print model predictions (optional)
    print(f"Model predictions: Covid={prob_covid:.2f}, Normal={prob_normal:.2f}, Pneumonia={prob_pneumonia:.2f}, TB={prob_tb:.2f}")

    # Set up fuzzy system
    diagnosis_sim = setup_fuzzy_system()

    # Provide inputs to fuzzy system
    diagnosis_sim.input['prob_covid'] = prob_covid
    diagnosis_sim.input['prob_normal'] = prob_normal
    diagnosis_sim.input['prob_pneumonia'] = prob_pneumonia
    diagnosis_sim.input['prob_tb'] = prob_tb
    
    # Compute the fuzzy result
    diagnosis_sim.compute()
    
    # Get the output diagnosis from the fuzzy system
    diagnosis_result = diagnosis_sim.output['diagnosis']
    
    if diagnosis_result < 0.33:
        diagnosis = 'Normal'
    elif 0.33 <= diagnosis_result < 0.67:
        diagnosis = 'Mild Disease'
    else:
        diagnosis = 'Severe Disease'
    
    # Display only the final diagnosis
    print(f"Final Diagnosis: {diagnosis}")
    
    return diagnosis

# Main function to load model and predict
def main():
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = load_model('best_model.pth', device)
    
    # Predict on a new X-ray image
    image_path = r"D:/images/Normal-7.png"
    predict_disease(image_path, model, device)

if __name__ == "__main__":
    main()
