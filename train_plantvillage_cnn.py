import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
try:
    from imblearn.keras import BalancedBatchGenerator
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from tensorflow.keras import mixed_precision
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlantDiseaseTrainer:
    def __init__(self, img_size=(224, 224), model_version='B3', use_mixed_precision=True, ensemble_size=1):
        self.train_dir = os.path.join('plantvillage_data', 'train')
        self.val_dir = os.path.join('plantvillage_data', 'validation')
        self.img_size = img_size  # Increased to 224x224
        self.batch_size = 32  # Reduced batch size
        self.epochs = 30
        self.learning_rate = 0.0001  # Lower learning rate
        self.class_weights = None
        self.model_version = model_version
        self.ensemble_size = ensemble_size
        self.use_mixed_precision = use_mixed_precision
        
        # Enable mixed precision if requested and supported
        if self.use_mixed_precision:
            try:
                mixed_precision.set_global_policy('mixed_float16')
                logger.info("Mixed precision training enabled.")
            except Exception as e:
                logger.warning(f"Mixed precision not enabled: {e}")
        
        # Create log directory
        self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create model directory
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def create_advanced_data_generators(self):
        """Create advanced data generators with reduced augmentation."""
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,  # Reduced
            width_shift_range=0.05,  # Reduced
            height_shift_range=0.05,  # Reduced
            shear_range=0.05,  # Reduced
            zoom_range=0.05,  # Reduced
            horizontal_flip=True,
            vertical_flip=False,  # Reduced
            fill_mode='nearest',
            brightness_range=[0.9, 1.1],  # Reduced
            channel_shift_range=10.0,  # Reduced
            validation_split=0.0
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Calculate class weights
        class_counts = train_generator.classes
        total_samples = len(class_counts)
        class_weights = {}
        for class_idx in range(len(train_generator.class_indices)):
            class_count = np.sum(class_counts == class_idx)
            class_weights[class_idx] = total_samples / (len(train_generator.class_indices) * class_count)
        
        self.class_weights = class_weights
        
        # Log class information
        logger.info(f"Found {len(train_generator.class_indices)} classes: {list(train_generator.class_indices.keys())}")
        logger.info("Class weights calculated:")
        for class_name, weight in zip(train_generator.class_indices.keys(), class_weights.values()):
            logger.info(f"{class_name}: {weight:.2f}")
        
        # Add oversampling if imblearn is available
        if IMBLEARN_AVAILABLE:
            logger.info("Using class-balanced oversampling with imblearn.")
            X = train_generator.filepaths
            y = train_generator.classes
            ros = RandomOverSampler()
            X_res, y_res = ros.fit_resample(np.array(X).reshape(-1, 1), y)
            self._oversampled_n_samples = len(X_res)
            self._oversampled_batch_size = self.batch_size
            def balanced_generator():
                while True:
                    idxs = np.random.choice(len(X_res), self.batch_size)
                    batch_x = [X_res[i][0] for i in idxs]
                    batch_y = [y_res[i] for i in idxs]
                    images = [img_to_array(load_img(f, target_size=self.img_size))/255.0 for f in batch_x]
                    yield np.stack(images), tf.keras.utils.to_categorical(batch_y, num_classes=len(train_generator.class_indices))
            train_gen = balanced_generator()
            return train_gen, validation_generator
        else:
            logger.info("imblearn not available, using standard generator.")
            return train_generator, validation_generator
    
    def create_enhanced_model(self):
        """Create an enhanced model architecture for better performance."""
        logger.info(f"Creating EfficientNet{self.model_version} model architecture...")
        
        # Select EfficientNet version
        if self.model_version == 'B3':
            base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        elif self.model_version == 'B4':
            base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        elif self.model_version == 'B5':
            base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        else:
            base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        
        # Unfreeze more layers for fine-tuning
        for layer in base_model.layers[:-60]:  # Unfreeze last 60 layers
            layer.trainable = False
        for layer in base_model.layers[-60:]:
            layer.trainable = True
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        predictions = Dense(len(self.class_weights), activation='softmax', dtype='float32')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model with advanced optimizer settings
        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def create_callbacks(self):
        """Create advanced callbacks for training."""
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Reduced patience
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,  # Reduced patience
            min_lr=1e-6,
            verbose=1
        )
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        return [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    def train(self):
        """Train the model with advanced techniques, optionally as an ensemble."""
        try:
            # Create data generators
            train_generator, validation_generator = self.create_advanced_data_generators()
            
            # Create callbacks
            callbacks = self.create_callbacks()
            
            histories = []
            models = []
            for i in range(self.ensemble_size):
                logger.info(f"Training model {i+1}/{self.ensemble_size}...")
                # Create model
                self.model = self.create_enhanced_model()
                
                # If using imblearn's BalancedBatchGenerator or custom oversampled generator, use stored values
                if hasattr(self, '_oversampled_n_samples') and hasattr(self, '_oversampled_batch_size'):
                    n_samples = self._oversampled_n_samples
                    batch_size = self._oversampled_batch_size
                # Fallback for DirectoryIterator
                elif hasattr(train_generator, 'n') and hasattr(train_generator, 'batch_size'):
                    n_samples = train_generator.n
                    batch_size = train_generator.batch_size
                else:
                    raise ValueError('Could not determine steps_per_epoch for the training generator. Please set n_samples and batch_size manually if using a custom generator.')

                steps_per_epoch = int(np.ceil(n_samples / batch_size))

                history = self.model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=self.epochs,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Save model
                self.model.save(os.path.join(self.model_dir, f'final_model_{i+1}.h5'))
                histories.append(history)
                models.append(self.model)
            
            # Save ensemble info
            if self.ensemble_size > 1:
                logger.info(f"Ensemble of {self.ensemble_size} models trained.")
            
            # Plot and evaluate last model
            self.plot_training_history(histories[-1])
            self.evaluate_model(validation_generator, models)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            raise
    
    def plot_training_history(self, history):
        """Plot and save training history."""
        # Create plots directory
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_history.png'))
        plt.close()
    
    def evaluate_model(self, validation_generator, models=None):
        """Evaluate model or ensemble performance."""
        if models is None:
            models = [self.model]
        # Get predictions from all models and average
        preds = []
        for m in models:
            preds.append(m.predict(validation_generator))
        predictions = np.mean(preds, axis=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = validation_generator.classes
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        logger.info(f"Final Validation Accuracy: {accuracy:.4f}")
        
        # Save evaluation results
        results = {
            'accuracy': accuracy,
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist()
        }
        
        # Save to CSV
        pd.DataFrame(results).to_csv(os.path.join(self.log_dir, 'evaluation_results.csv'), index=False)
    
    def prepare_debug_subset(self, subset_classes, images_per_class=100):
        """Create a small debug subset with a few classes and images for overfitting test."""
        debug_dir = 'plantvillage_data_debug'
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        for class_name in subset_classes:
            src_dir = os.path.join(self.train_dir, class_name)
            dst_dir = os.path.join(debug_dir, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in images[:images_per_class]:
                shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
        return debug_dir

    def train_debug_subset(self, subset_classes, images_per_class=100):
        """Train on a small subset for overfitting/debugging."""
        debug_dir = self.prepare_debug_subset(subset_classes, images_per_class)
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            debug_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        model = self.create_enhanced_model()
        model.fit(
            train_generator,
            epochs=20,
            verbose=2
        )
        shutil.rmtree(debug_dir)

def main():
    # Example: Use EfficientNetB3, 224x224, mixed precision, ensemble of 1
    trainer = PlantDiseaseTrainer(img_size=(224,224), model_version='B3', use_mixed_precision=True, ensemble_size=1)
    trainer.train()

    # Debug: Overfit on a small subset (uncomment to use)
    trainer = PlantDiseaseTrainer(img_size=(224,224), model_version='B3', use_mixed_precision=True, ensemble_size=1)
    trainer.train_debug_subset(['Tomato_healthy', 'Tomato_Bacterial_spot', 'Pepper__bell___healthy'], images_per_class=100)

if __name__ == "__main__":
    main()
