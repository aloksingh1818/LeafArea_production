import os
import shutil
import logging
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    def __init__(self, source_dir, target_dir, test_size=0.2, min_image_size=(100, 100)):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.test_size = test_size
        self.min_image_size = min_image_size
        self.valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'class_distribution': {},
            'image_sizes': [],
            'corrupted_files': []
        }

    def validate_image(self, image_path):
        """Validate image quality and format."""
        try:
            # Check file extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.valid_extensions:
                return False, "Invalid file extension"

            # Try to open image
            with Image.open(image_path) as img:
                # Check image size
                if img.size[0] < self.min_image_size[0] or img.size[1] < self.min_image_size[1]:
                    return False, f"Image too small: {img.size}"

                # Check if image is corrupted
                img.verify()
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Check image quality
                img_array = np.array(img)
                if img_array.std() < 10:  # Check if image is too uniform
                    return False, "Image too uniform (possible blank image)"
                
                # Store image size
                self.stats['image_sizes'].append(img.size)
                
                return True, "Valid image"

        except Exception as e:
            return False, f"Error processing image: {str(e)}"

    def organize_dataset(self):
        """Organize dataset into train and validation sets."""
        logger.info("Starting dataset organization...")
        
        # Check if source directory exists
        if not os.path.exists(self.source_dir):
            logger.error(f"Source directory not found: {self.source_dir}")
            logger.info("Please create a directory named 'raw_dataset' and place your plant disease images in it.")
            logger.info("The directory structure should be:")
            logger.info("raw_dataset/")
            logger.info("    Pepper__bell___Bacterial_spot/")
            logger.info("        image1.jpg")
            logger.info("        image2.jpg")
            logger.info("    Pepper__bell___healthy/")
            logger.info("        image1.jpg")
            logger.info("        image2.jpg")
            logger.info("    ...")
            return
        
        # Create target directories
        train_dir = os.path.join(self.target_dir, 'train')
        val_dir = os.path.join(self.target_dir, 'validation')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Process each class directory
        class_dirs = [d for d in os.listdir(self.source_dir) if os.path.isdir(os.path.join(self.source_dir, d))]
        
        if not class_dirs:
            logger.error(f"No class directories found in {self.source_dir}")
            logger.info("Please make sure your raw_dataset directory contains subdirectories for each plant disease class.")
            return

        for class_name in class_dirs:
            class_dir = os.path.join(self.source_dir, class_name)
            logger.info(f"Processing class: {class_name}")
            self.stats['class_distribution'][class_name] = 0

            # Get all images in class directory
            image_files = []
            for file in os.listdir(class_dir):
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    image_files.append(os.path.join(class_dir, file))

            if not image_files:
                logger.warning(f"No valid images found in {class_dir}")
                continue

            # Split into train and validation
            train_files, val_files = train_test_split(
                image_files, 
                test_size=self.test_size, 
                random_state=42
            )

            # Process training images
            for file in tqdm(train_files, desc=f"Processing {class_name} training images"):
                self._process_image(file, train_dir, class_name)

            # Process validation images
            for file in tqdm(val_files, desc=f"Processing {class_name} validation images"):
                self._process_image(file, val_dir, class_name)

        self._generate_report()

    def _process_image(self, file_path, target_dir, class_name):
        """Process and copy a single image."""
        self.stats['total_images'] += 1
        
        # Validate image
        is_valid, message = self.validate_image(file_path)
        
        if is_valid:
            # Create class directory if it doesn't exist
            class_target_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_target_dir, exist_ok=True)
            
            # Copy and resize image
            target_path = os.path.join(class_target_dir, os.path.basename(file_path))
            self._copy_and_resize_image(file_path, target_path)
            
            self.stats['valid_images'] += 1
            self.stats['class_distribution'][class_name] += 1
        else:
            self.stats['invalid_images'] += 1
            self.stats['corrupted_files'].append((file_path, message))
            logger.warning(f"Invalid image {file_path}: {message}")

    def _copy_and_resize_image(self, source_path, target_path):
        """Copy and resize image to target path."""
        try:
            with Image.open(source_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image maintaining aspect ratio
                img.thumbnail((128, 128), Image.Resampling.LANCZOS)
                
                # Save resized image
                img.save(target_path, quality=95, optimize=True)
        except Exception as e:
            logger.error(f"Error processing image {source_path}: {str(e)}")

    def _generate_report(self):
        """Generate a detailed report of the dataset preparation."""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images_processed': self.stats['total_images'],
            'valid_images': self.stats['valid_images'],
            'invalid_images': self.stats['invalid_images'],
            'class_distribution': self.stats['class_distribution'],
            'average_image_size': np.mean(self.stats['image_sizes'], axis=0).tolist() if self.stats['image_sizes'] else [0, 0],
            'corrupted_files': self.stats['corrupted_files']
        }

        # Save report to CSV
        df = pd.DataFrame([report])
        df.to_csv('dataset_preparation_report.csv', index=False)
        
        # Log summary
        logger.info("\nDataset Preparation Summary:")
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Valid images: {self.stats['valid_images']}")
        logger.info(f"Invalid images: {self.stats['invalid_images']}")
        logger.info("\nClass Distribution:")
        for class_name, count in self.stats['class_distribution'].items():
            logger.info(f"{class_name}: {count} images")
        
        if self.stats['corrupted_files']:
            logger.warning("\nCorrupted Files:")
            for file, reason in self.stats['corrupted_files']:
                logger.warning(f"{file}: {reason}")

def main():
    # Configuration
    source_dir = os.path.join(os.getcwd(), 'raw_dataset')  # Your original dataset directory
    target_dir = os.path.join(os.getcwd(), 'plantvillage_data')  # Where to save organized dataset
    
    # Create preparator instance
    preparator = DatasetPreparator(source_dir, target_dir)
    
    # Organize dataset
    preparator.organize_dataset()

if __name__ == "__main__":
    main() 