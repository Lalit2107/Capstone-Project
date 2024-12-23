import os
import cv2  # For OpenCV functions
import shutil
from tqdm import tqdm  # To show progress bar

# Function to apply histogram equalization and save the enhanced image
def apply_histogram_equalization(image_path, output_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)
    
    # Save the equalized image to the output path
    cv2.imwrite(output_path, equalized_image)

# Function to process a dataset (train, test, val)
def process_dataset(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through each subfolder (covid, normal, pneumonia, tb)
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        output_subfolder_path = os.path.join(output_folder, subfolder)
        
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)
        
        # Iterate over each image in the subfolder
        for image_name in tqdm(os.listdir(subfolder_path), desc=f"Processing {subfolder}"):
            image_path = os.path.join(subfolder_path, image_name)
            output_image_path = os.path.join(output_subfolder_path, image_name)
            
            # Apply histogram equalization and save the enhanced image
            apply_histogram_equalization(image_path, output_image_path)

# Main function to apply histogram equalization to all datasets
def main():
    base_input_dir = r"D:/Dataset_split"  # Replace with the directory containing your split dataset
    base_output_dir = r"D:/Enhanced_Dataset_Split"  # Output folder for enhanced dataset
    
    # Process train, test, and validation datasets
    train_folder = os.path.join(base_input_dir, 'train')
    test_folder = os.path.join(base_input_dir, 'test')
    val_folder = os.path.join(base_input_dir, 'val')

    enhanced_train_folder = os.path.join(base_output_dir, 'train')
    enhanced_test_folder = os.path.join(base_output_dir, 'test')
    enhanced_val_folder = os.path.join(base_output_dir, 'val')

    # Apply histogram equalization to each dataset
    print("Processing train dataset...")
    process_dataset(train_folder, enhanced_train_folder)

    print("Processing test dataset...")
    process_dataset(test_folder, enhanced_test_folder)

    print("Processing validation dataset...")
    process_dataset(val_folder, enhanced_val_folder)

if __name__ == "__main__":
    main()
