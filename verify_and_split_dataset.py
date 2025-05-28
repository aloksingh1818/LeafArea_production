import os
import shutil
import logging
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import cv2
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetVerifier:
    def __init__(self):
        self.base_dir = os.path.join(os.getcwd(), 'plantvillage_data')
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.val_dir = os.path.join(self.base_dir, 'validation')
        self.min_image_size = (5, 5)  # Even more lenient minimum size
        self.max_image_size = (10000, 10000)  # Even more lenient maximum size
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        self.target_size = (128, 128)
        self.batch_size = 100
        
    def verify_image(self, image_path):
        """Verify if an image is valid and meets quality standards."""
        try:
            # Quick check for file extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.valid_extensions:
                logger.debug(f"Invalid extension for {image_path}: {ext}")
                return False, "Invalid file extension"
            
            # Open and verify image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Check image size
                width, height = img.size
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    logger.debug(f"Image too small: {image_path} ({width}x{height})")
                    return False, f"Image too small: {width}x{height}"
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    logger.debug(f"Image too large: {image_path} ({width}x{height})")
                    return False, f"Image too large: {width}x{height}"
                
                # Basic quality check
                img_array = np.array(img)
                mean_value = img_array.mean()
                if mean_value < 0.5 or mean_value > 254.5:  # More lenient thresholds
                    logger.debug(f"Image too dark/bright: {image_path} (mean: {mean_value})")
                    return False, f"Image too dark or too bright (mean: {mean_value})"
                
                return True, "Valid image"
                
        except Exception as e:
            logger.debug(f"Error processing {image_path}: {str(e)}")
            return False, f"Error processing image: {str(e)}"
    
    def process_batch(self, image_paths, desc=""):
        """Process a batch of images."""
        results = []
        for img_path in tqdm(image_paths, desc=desc, leave=False):
            if os.path.isfile(img_path):
                is_valid, message = self.verify_image(img_path)
                results.append((img_path, is_valid, message))
        return results
    
    def analyze_dataset(self):
        """Analyze the dataset and return statistics."""
        stats = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'class_distribution': {},
            'image_sizes': [],
            'corrupted_images': [],
            'invalid_sizes': [],
            'invalid_modes': [],
            'invalid_extensions': [],
            'too_dark_bright': []
        }
        
        image_info = []
        start_time = time.time()
        
        logger.info("Starting dataset analysis...")
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.base_dir) 
                     if os.path.isdir(os.path.join(self.base_dir, d)) 
                     and d not in ['train', 'validation']]
        
        # Process each class
        for class_name in tqdm(class_dirs, desc="Processing classes"):
            class_dir = os.path.join(self.base_dir, class_name)
            image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                         if os.path.isfile(os.path.join(class_dir, f))]
            
            # Process images in batches
            for i in range(0, len(image_paths), self.batch_size):
                batch = image_paths[i:i + self.batch_size]
                results = self.process_batch(batch, f"Processing {class_name}")
                
                for img_path, is_valid, message in results:
                    stats['total_images'] += 1
                    
                    if is_valid:
                        stats['valid_images'] += 1
                        with Image.open(img_path) as img:
                            stats['image_sizes'].append(img.size)
                    else:
                        stats['invalid_images'] += 1
                        if "corrupted" in message.lower():
                            stats['corrupted_images'].append(img_path)
                        elif "size" in message.lower():
                            stats['invalid_sizes'].append(img_path)
                        elif "mode" in message.lower():
                            stats['invalid_modes'].append(img_path)
                        elif "extension" in message.lower():
                            stats['invalid_extensions'].append(img_path)
                        elif "dark" in message.lower() or "bright" in message.lower():
                            stats['too_dark_bright'].append(img_path)
                    
                    image_info.append({
                        'class': class_name,
                        'filename': os.path.basename(img_path),
                        'path': img_path,
                        'is_valid': is_valid,
                        'message': message
                    })
            
            # Update class distribution
            valid_count = sum(1 for info in image_info if info['class'] == class_name and info['is_valid'])
            stats['class_distribution'][class_name] = valid_count
            
            # Log progress with more details
            elapsed_time = time.time() - start_time
            logger.info(f"Processed {class_name}: {valid_count} valid images out of {len(image_paths)} "
                       f"({elapsed_time:.1f} seconds elapsed)")
            
            # Log some example invalid images for debugging
            if valid_count == 0:
                invalid_images = [info for info in image_info if info['class'] == class_name and not info['is_valid']]
                if invalid_images:
                    logger.info(f"Example invalid images in {class_name}:")
                    for info in invalid_images[:3]:
                        logger.info(f"- {info['filename']}: {info['message']}")
        
        # Save results
        df = pd.DataFrame(image_info)
        df.to_csv('dataset_analysis.csv', index=False)
        
        return stats, df
    
    def split_dataset(self, test_size=0.2, random_state=42):
        """Split the dataset into train and validation sets."""
        logger.info("Splitting dataset into train and validation sets...")
        
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        
        for class_name in os.listdir(self.base_dir):
            class_dir = os.path.join(self.base_dir, class_name)
            if not os.path.isdir(class_dir) or class_name in ['train', 'validation']:
                continue
            
            # Get valid images
            valid_images = []
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    is_valid, _ = self.verify_image(img_path)
                    if is_valid:
                        valid_images.append(img_path)
            
            if not valid_images:
                logger.warning(f"No valid images found for class {class_name}")
                continue
            
            # Adjust test size if needed
            if len(valid_images) < 5:
                test_size = 0.0  # Use all images for training if very few valid images
            
            # Split and process
            train_images, val_images = train_test_split(
                valid_images, test_size=test_size, random_state=random_state
            )
            
            # Create directories
            train_class_dir = os.path.join(self.train_dir, class_name)
            val_class_dir = os.path.join(self.val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            
            # Process in batches
            for batch in [train_images[i:i + self.batch_size] 
                         for i in range(0, len(train_images), self.batch_size)]:
                for img_path in tqdm(batch, desc=f"Processing {class_name} - Train", leave=False):
                    self._copy_and_resize_image(img_path, train_class_dir)
            
            for batch in [val_images[i:i + self.batch_size] 
                         for i in range(0, len(val_images), self.batch_size)]:
                for img_path in tqdm(batch, desc=f"Processing {class_name} - Validation", leave=False):
                    self._copy_and_resize_image(img_path, val_class_dir)
    
    def _copy_and_resize_image(self, src_path, dst_dir):
        """Copy and resize an image to the target directory."""
        try:
            img = cv2.imread(src_path)
            if img is None:
                logger.error(f"Could not read image: {src_path}")
                return
            
            img = cv2.resize(img, self.target_size)
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            cv2.imwrite(dst_path, img)
            
        except Exception as e:
            logger.error(f"Error processing {src_path}: {str(e)}")
    
    def generate_report(self, stats, df):
        """Generate a detailed report of the dataset analysis."""
        report = []
        report.append("Dataset Analysis Report")
        report.append("=" * 50)
        report.append(f"\nTotal Images: {stats['total_images']}")
        report.append(f"Valid Images: {stats['valid_images']}")
        report.append(f"Invalid Images: {stats['invalid_images']}")
        
        report.append("\nClass Distribution:")
        for class_name, count in stats['class_distribution'].items():
            report.append(f"- {class_name}: {count} images")
        
        # Add error summaries
        for error_type, error_list in [
            ("Corrupted Images", stats['corrupted_images']),
            ("Invalid Sizes", stats['invalid_sizes']),
            ("Invalid Modes", stats['invalid_modes']),
            ("Invalid Extensions", stats['invalid_extensions']),
            ("Too Dark/Bright", stats['too_dark_bright'])
        ]:
            if error_list:
                report.append(f"\n{error_type}:")
                for img in error_list[:5]:
                    report.append(f"- {img}")
                if len(error_list) > 5:
                    report.append(f"... and {len(error_list) - 5} more")
        
        # Save report
        with open('dataset_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Save summary
        summary = df.groupby('class').agg({
            'is_valid': ['count', 'sum'],
            'message': lambda x: x.value_counts().to_dict()
        }).round(2)
        summary.to_csv('dataset_summary.csv')
        
        return '\n'.join(report)

def main():
    verifier = DatasetVerifier()
    
    try:
        start_time = time.time()
        
        # Analyze dataset
        stats, df = verifier.analyze_dataset()
        
        # Generate report
        report = verifier.generate_report(stats, df)
        logger.info("\n" + report)
        
        # Split dataset
        verifier.split_dataset()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nDataset processing completed in {elapsed_time:.1f} seconds!")
        logger.info("Check the following files for detailed information:")
        logger.info("- dataset_analysis.csv: Detailed analysis of each image")
        logger.info("- dataset_summary.csv: Summary statistics by class")
        logger.info("- dataset_analysis_report.txt: Human-readable report")
        logger.info("- dataset_verification.log: Processing log")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 