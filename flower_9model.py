import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.applications import (
    MobileNetV2, EfficientNetB0, InceptionV3, VGG19, ResNet50, DenseNet121, Xception,
    EfficientNetB3, ResNet101, InceptionResNetV2, VGG16 
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

base_path = 'flower_images_split' 
output_dir = 'flower_images_output'
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Base dataset path: {base_path}")
logging.info(f"Output directory: {output_dir}")

IMG_WIDTH, IMG_HEIGHT = 224, 224
CHANNEL_NUM = 3
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 100 

input_shape = (IMG_WIDTH, IMG_HEIGHT, CHANNEL_NUM)
logging.info(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}x{CHANNEL_NUM}")
logging.info(f"Batch size: {BATCH_SIZE}")
logging.info(f"Training epochs per model: {EPOCHS}")

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.info("\n--- Initial Dataset Overview (from folder structure) ---")
split_totals = {}

if not os.path.isdir(base_path):
    logging.error(f"Error: Base path '{base_path}' does not exist. Please check your Drive path.")
    exit()
else:
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.isdir(split_path):
            classes = [name for name in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, name))]
            total_images = sum(len([f for f in os.listdir(os.path.join(split_path, class_name))
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                               for class_name in classes)
            split_totals[split] = {'classes': len(classes), 'images': total_images}
        else:
            logging.warning(f"Warning: Split folder '{split_path}' not found. Skipping this split.")

    for split, data in split_totals.items():
        logging.info(f"{split.upper()} Split:")
        logging.info(f"  Number of classes: {data['classes']}")
        logging.info(f"  Total images: {data['images']}")
        logging.info("")

class_counts_overall = {}
if os.path.isdir(base_path):
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split_name)
        if os.path.isdir(split_path):
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    class_counts_overall[class_name] = class_counts_overall.get(class_name, 0) + image_count

if class_counts_overall:
    min_count_overall = min(class_counts_overall.values()) if class_counts_overall else 0
    max_count_overall = max(class_counts_overall.values()) if class_counts_overall else 0
    class_counts_overall = dict(sorted(class_counts_overall.items(), key=lambda item: item[1], reverse=True))
    logging.info(f"Total number of classes (overall): {len(class_counts_overall)}")
    logging.info(f"Total number of images (overall): {sum(class_counts_overall.values())}")
    logging.info("Class-wise image counts (overall):")
    for class_name, count in class_counts_overall.items():
        logging.info(f"  {class_name}: {count} images")
    logging.info(f"\nMinimum number of images in a class (overall): {min_count_overall}")
    logging.info(f"Maximum number of images in a class (overall): {max_count_overall}")
else:
    logging.warning(f"No class folders found within the split directories under '{base_path}'.")

logging.info("\n--- Defining Callbacks ---")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch < 10:
        return lr * 0.5
    elif epoch < 15:
        return lr * 0.2
    else:
        return lr * 0.1

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

logging.info("Callbacks defined: EarlyStopping, LearningRateScheduler (ModelCheckpoint will be dynamic)")

def create_dataframe_from_folder(base_dir):
    filepaths = []
    labels = []
    data_sets = []

    for split_name in ['train', 'test', 'val']:
        split_path = os.path.join(base_dir, split_name)
        if not os.path.isdir(split_path):
            logging.warning(f"Warning: Split folder '{split_path}' not found. Skipping this split.")
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        full_path = os.path.join(class_path, file)
                        relative_path = os.path.relpath(full_path, base_dir)

                        filepaths.append(relative_path)
                        labels.append(class_name)
                        data_sets.append(split_name)

    new_df = pd.DataFrame({
        'filepaths': filepaths,
        'labels': labels,
        'image_path': [os.path.join(base_dir, fp) for fp in filepaths],
        'data set': data_sets
    })
    return new_df

logging.info(f"\n--- Scanning '{base_path}' to create initial DataFrame from folder structure ---")
df = create_dataframe_from_folder(base_path)

if df.empty:
    raise ValueError(f"No image files found in '{base_path}'. Please check your base_path and folder structure.")
logging.info(f"DataFrame created with {len(df)} entries.")

train_df_original = df[df['data set'] == 'train'].copy()
test_df_original = df[df['data set'] == 'test'].copy()
validation_df_original = df[df['data set'] == 'val'].copy()

train_df_original = train_df_original.rename(columns={'labels': 'label'})
test_df_original = test_df_original.rename(columns={'labels': 'label'})
validation_df_original = validation_df_original.rename(columns={'labels': 'label'})
logging.info("DataFrames for train, validation, and test splits created and columns renamed.")

logging.info("\n--- Detailed Data Distribution (Text Summary) ---")

logging.info("\nTRAIN Split:")
train_class_distribution = train_df_original['label'].value_counts().sort_index()
logging.info(f"Number of classes: {len(train_class_distribution)}")
logging.info(f"Total images: {len(train_df_original)}")
logging.info("Class-wise image counts:")
for class_name, count in train_class_distribution.items():
    logging.info(f"  {class_name}: {count} images")
logging.info(f"Minimum images per class: {train_class_distribution.min()} images")
logging.info(f"Maximum images per class: {train_class_distribution.max()} images")

logging.info("\nVAL Split:")
val_class_distribution = validation_df_original['label'].value_counts().sort_index()
logging.info(f"Number of classes: {len(val_class_distribution)}")
logging.info(f"Total images: {len(validation_df_original)}")
logging.info("Class-wise image counts:")
for class_name, count in val_class_distribution.items():
    logging.info(f"  {class_name}: {count} images")
logging.info(f"Minimum images per class: {val_class_distribution.min()} images")
logging.info(f"Maximum images per class: {val_class_distribution.max()} images")

logging.info("\nTEST Split:")
test_class_distribution = test_df_original['label'].value_counts().sort_index()
logging.info(f"Number of classes: {len(test_class_distribution)}")
logging.info(f"Total images: {len(test_df_original)}")
logging.info("Class-wise image counts:")
for class_name, count in test_class_distribution.items():
    logging.info(f"  {class_name}: {count} images")
logging.info(f"Minimum images per class: {test_class_distribution.min()} images")
logging.info(f"Maximum images per class: {test_class_distribution.max()} images")

def create_generators(train_df, val_df, test_df, img_height, img_width, batch_size, random_seed):
    logging.info("\n--- Setting up ImageDataGenerators ---")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=random_seed
    )

    validation_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    logging.info(f"Detected NUM_CLASSES from generators: {num_classes}")

    logging.info("\n--- Class Serialization (Class Name to Index Mapping) ---")
    sorted_class_indices = sorted(train_generator.class_indices.items(), key=lambda item: item[1])
    class_names = [item[0] for item in sorted_class_indices]
    for class_name, index in sorted_class_indices:
        logging.info(f"  {class_name}: {index}")
    
    return train_generator, validation_generator, test_generator, num_classes, class_names

def plot_overall_class_distribution(df_combined):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_combined['labels'].value_counts().index, y=df_combined['labels'].value_counts().values,
                hue=df_combined['labels'].value_counts().index, palette='cubehelix', legend=False)
    plt.title('Overall Distribution of Classes Across All Data (Train, Test, Val Combined)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_split_class_distribution(df_split, split_name, palette_name):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_split['label'].value_counts().index, y=df_split['label'].value_counts().values,
                hue=df_split['label'].value_counts().index, palette=palette_name, legend=False)
    plt.title(f'Distribution of Classes in {split_name} Set')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

logging.info("\n--- Plotting Overall Class Distribution ---")
plot_overall_class_distribution(df)

logging.info("\n--- Plotting Training Set Class Distribution ---")
plot_split_class_distribution(train_df_original, 'Training', 'viridis')

logging.info("\n--- Plotting Validation Set Class Distribution ---")
plot_split_class_distribution(validation_df_original, 'Validation', 'magma')


def build_model(base_model_class, model_name, num_classes, input_shape):
    logging.info(f"Building model for {model_name} with input_shape: {input_shape} and {num_classes} classes.")
    
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')

    base_model.trainable = False
    logging.info(f"{model_name} base model loaded and frozen.")

    x = base_model.output
    x = Dense(1024, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    logging.info("Custom classification head added.")
    return model, base_model

def train_and_evaluate_model(model, model_name, train_gen, val_gen, test_gen, class_names, epochs, callbacks_list, output_dir):
    """
    Compiles, trains, evaluates, and plots results for a given model.
    """
    logging.info(f"\n--- Compiling {model_name} Model ---")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    logging.info(f"{model_name} Model compiled successfully.")

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, f'flower_{model_name}.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    current_callbacks = callbacks_list + [model_checkpoint]

    logging.info(f"\n--- Starting {model_name} Model Training (Head Only, Base Frozen) ---")
    history = None
    try:
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=current_callbacks,
            verbose=1
        )
        logging.info(f"\n--- {model_name} Model Training Finished ---")
    except Exception as e:
        logging.error(f"Error during {model_name} model training: {e}")

    test_loss, test_accuracy = None, None
    logging.info(f"\n--- Evaluating {model_name} Model on Test Set ---")
    if history:
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
        logging.info(f"{model_name} Test Loss: {test_loss:.4f}")
        logging.info(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    else:
        logging.warning(f"{model_name} model training failed, skipping evaluation.")

    if history:
        logging.info(f"\n--- Generating Performance Plots for {model_name} ---")
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        logging.info(f"\n--- Generating Confusion Matrix for {model_name} ---")
        test_gen.reset() 
        Y_pred = model.predict(test_gen, steps=len(test_gen), verbose=1)
        y_pred_classes = np.argmax(Y_pred, axis=1)

        y_true = test_gen.classes

        conf_matrix = confusion_matrix(y_true, y_pred_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        logging.info(f"\n--- Generating Classification Report for {model_name} ---")
        report = classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0)
        logging.info(report)
    else:
        logging.warning(f"Skipping performance plots for {model_name} as training did not complete successfully.")
    
    return history, test_accuracy

def plot_model_comparison(model_results):
    logging.info("\n--- Plotting Model Comparison ---")
    model_names = []
    test_accuracies = []

    for name, results in model_results.items():
        if results['test_accuracy'] is not None:
            model_names.append(name)
            test_accuracies.append(results['test_accuracy'])
    
    if not model_names:
        logging.warning("No models had successful test accuracy to compare.")
        return

    plt.figure(figsize=(12, 7))
    sns.barplot(x=model_names, y=test_accuracies, palette='deep')
    plt.title('Comparison of Test Accuracies Across Different Models')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1) 
    plt.xticks(rotation=45, ha='right')
    for index, value in enumerate(test_accuracies):
        plt.text(index, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def main():
    train_generator, validation_generator, test_generator, num_classes, class_names = \
        create_generators(train_df_original, validation_df_original, test_df_original, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, RANDOM_SEED)

    models_to_train = {
        'MobileNetV2': MobileNetV2,
        'InceptionV3': InceptionV3,
        'VGG16': VGG16, 
        'VGG19': VGG19, 
        'ResNet50': ResNet50,
        'ResNet101': ResNet101, 
        'DenseNet121': DenseNet121,
        'Xception': Xception,
        'InceptionResNetV2': InceptionResNetV2 
    }

    model_results = {}

    for model_name, model_fn in models_to_train.items():
        logging.info(f"\n{'='*80}")
        logging.info(f"STARTING TRAINING FOR MODEL: {model_name}")
        logging.info(f"{'='*80}")

        tf.keras.backend.clear_session() 
        
        model, _ = build_model(model_fn, model_name, num_classes, input_shape)
        
        history, test_accuracy = train_and_evaluate_model(
            model, model_name, train_generator, validation_generator, 
            test_generator, class_names, EPOCHS, [early_stopping, lr_scheduler], output_dir
        )
        
        model_results[model_name] = {
            'history': history,
            'test_accuracy': test_accuracy
        }
        logging.info(f"\n{'='*80}")
        logging.info(f"FINISHED TRAINING FOR MODEL: {model_name}")
        logging.info(f"{'='*80}\n")

    plot_model_comparison(model_results)

if __name__ == "__main__":
    main()
