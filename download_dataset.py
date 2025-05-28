import os
import logging
from tqdm import tqdm
import shutil
import subprocess
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self):
        self.raw_dataset_dir = os.path.join(os.getcwd(), 'raw_dataset')
        self.temp_dir = os.path.join(os.getcwd(), 'temp_download')
        self.kaggle_json_path = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
        
    def check_kaggle_credentials(self):
        """Check if Kaggle credentials are properly set up."""
        if not os.path.exists(self.kaggle_json_path):
            logger.error("Kaggle credentials not found!")
            logger.info("Please follow these steps:")
            logger.info("1. Go to https://www.kaggle.com/account")
            logger.info("2. Click on 'Create New API Token'")
            logger.info("3. Save the kaggle.json file to ~/.kaggle/")
            logger.info("4. Make sure the file has the correct permissions")
            return False
        return True
        
    def download_dataset(self):
        """Download the PlantVillage dataset using Kaggle API."""
        logger.info("Starting dataset download...")
        
        if not self.check_kaggle_credentials():
            return
        
        # Create directories
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.raw_dataset_dir, exist_ok=True)
        
        try:
            # Change to temp directory
            os.chdir(self.temp_dir)
            
            # Download dataset using Kaggle API
            logger.info("Downloading PlantVillage dataset using Kaggle API...")
            subprocess.run([
                'kaggle', 'datasets', 'download',
                'abdallahalidev/plantvillage-dataset',
                '--unzip'
            ], check=True)
            
            # Move files to raw_dataset directory
            logger.info("Organizing dataset...")
            extracted_dir = os.path.join(self.temp_dir, 'PlantVillage')
            if not os.path.exists(extracted_dir):
                # Try to find the correct directory
                for root, dirs, files in os.walk(self.temp_dir):
                    if 'PlantVillage' in dirs:
                        extracted_dir = os.path.join(root, 'PlantVillage')
                        break
            
            if os.path.exists(extracted_dir):
                for item in os.listdir(extracted_dir):
                    src = os.path.join(extracted_dir, item)
                    dst = os.path.join(self.raw_dataset_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                raise Exception("Could not find PlantVillage directory in extracted files")
            
            logger.info("Dataset download and organization completed successfully!")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            logger.info("Please make sure you have the Kaggle API installed:")
            logger.info("pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            logger.info("Cleaning up temporary files...")
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            # Change back to original directory
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    def verify_dataset(self):
        """Verify that the dataset was downloaded and organized correctly."""
        if not os.path.exists(self.raw_dataset_dir):
            logger.error("Dataset directory not found!")
            return False
        
        class_dirs = [d for d in os.listdir(self.raw_dataset_dir) 
                     if os.path.isdir(os.path.join(self.raw_dataset_dir, d))]
        
        if not class_dirs:
            logger.error("No class directories found in dataset!")
            return False
        
        logger.info(f"Found {len(class_dirs)} plant disease classes:")
        for class_name in class_dirs:
            class_dir = os.path.join(self.raw_dataset_dir, class_name)
            image_count = len([f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            logger.info(f"- {class_name}: {image_count} images")
        
        return True

def main():
    downloader = DatasetDownloader()
    
    try:
        # Download and organize the dataset
        downloader.download_dataset()
        
        # Verify the dataset
        if downloader.verify_dataset():
            logger.info("\nDataset is ready for processing!")
            logger.info("You can now run: python prepare_dataset.py")
        else:
            logger.error("Dataset verification failed!")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.info("Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 