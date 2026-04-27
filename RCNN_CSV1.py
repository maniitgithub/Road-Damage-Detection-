import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
# Function to plot training history
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd


# Function to preprocess an image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    #img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 224.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to create and compile the object detection model
def create_object_detection_model(num_classes):

    # Specify the local path to the ResNet50 weights file
    local_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    base_model = ResNet50(weights=local_weights_path, include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    bbox_output = layers.Dense(4, name='bbox_output')(x)

    model = models.Model(inputs=base_model.input, outputs=[class_output, bbox_output])

    model.compile(optimizer='adam',
                  loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': 'mse'},
                  metrics={'class_output': 'accuracy', 'bbox_output': 'mae'})

    return model


# Function to create and compile the object detection model
def create_object_detection_model1(num_classes):

    # Specify the local path to the ResNet50 weights file
    local_weights_path = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False,weights=local_weights_path)

    for layer in base_model.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    bbox_output = layers.Dense(4, name='bbox_output')(x)

    model = models.Model(inputs=base_model.input, outputs=[class_output, bbox_output])

    model.compile(optimizer='adam',
                  loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': 'mse'},
                  metrics={'class_output': 'accuracy', 'bbox_output': 'mae'})

    return model

print("Start Process")


# Load and preprocess the dataset
csv_path = 'annotations.csv'
images_folder = 'potholes'

df = pd.read_csv(csv_path)
classes=df['class'].unique()
class_id_mapping = {cls: idx for idx, cls in enumerate(classes)}
df['class_id'] = df['class'].map(class_id_mapping)

class_id_to_class = {idx: cls for cls, idx in class_id_mapping.items()}
# Display the dictionary
print(class_id_to_class)

classesid=df['class_id'].unique()
num_classes = len(classesid)

# Get unique classes and their counts
class_counts = df['class'].value_counts()

# Plot a bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
plt.savefig("1.png")


print("Show Graph")

# Parse the label files and preprocess images
images = []
class_labels = []
bbox_labels = []
imgcc=0

# Loop through DataFrame rows
for _, row in df.iterrows():
    imgcc=imgcc+1
    image_path = os.path.join(images_folder, row['filename'])
    #image = img_to_array(load_img(image_path, target_size=(224, 224)))
    #images.append(image)
    images.append(preprocess_image(image_path))

    label = row['class_id']

    # Normalize bounding box coordinates
    xmin = row['xmin'] / 224.0
    ymin = row['ymin'] / 224.0
    xmax = row['xmax'] / 224.0
    ymax = row['ymax'] / 224.0     

    class_labels.append(label)
    bbox_labels.append([xmin, ymin, xmax, ymax])
    if imgcc==100:
        pass;
    

images = np.concatenate(images, axis=0)
class_labels = np.array(class_labels)
bbox_labels = np.array(bbox_labels)

print(len(images))
print(len(class_labels))
print(len(bbox_labels))

print("start modeling")
# Create and compile the model
num_classes = len(set(class_labels))
print(num_classes)
model = create_object_detection_model1(num_classes)

# Train the model
#model.fit(images, {'class_output': class_labels, 'bbox_output': bbox_labels}, epochs=10, batch_size=32)
history = model.fit(images, {'class_output': class_labels, 'bbox_output': bbox_labels}, epochs=10, batch_size=32)

# Save the model
model.save('object_detection_model.h5')


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['class_output_accuracy'], label='Class Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['class_output_loss'], label='Class Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

print("End")
