import os
import shutil
from sklearn.model_selection import train_test_split

# Set the dataset directory
data_dir = r"D:/Dataset_Chest"
output_dir = r"D:/Dataset_split"

# Create subfolders for train, test, and val splits
splits = ['train', 'test', 'val']
categories = ['covid', 'normal', 'pneumonia', 'tb']

for split in splits:
    for category in categories:
        path = os.path.join(output_dir, split, category)
        os.makedirs(path, exist_ok=True)

# Define train-test-validation split ratios
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

# Function to split dataset into train, test, val
def split_and_move_files(category):
    category_dir = os.path.join(data_dir, category)
    images = os.listdir(category_dir)
    
    # Split dataset into train, test, val
    train_images, temp_images = train_test_split(images, test_size=(test_ratio + val_ratio))
    val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (test_ratio + val_ratio)))

    # Move images to respective folders
    for img in train_images:
        shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'train', category, img))
        
    for img in test_images:
        shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'test', category, img))
        
    for img in val_images:
        shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'val', category, img))

# Split and move files for each category
for category in categories:
    split_and_move_files(category)

print("Dataset successfully split into train, test, and validation sets.")
