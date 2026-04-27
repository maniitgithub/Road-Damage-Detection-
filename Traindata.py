import tkinter as tk
import cv2
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import numpy as np
import os
#Function to plot training history
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd

def Train_Start():


    # Load and preprocess the dataset
    csv_path = 'annotations.csv'
    images_folder = 'potholes'
    
    images = []
    class_labels = []
    bbox_labels = []

    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    
    history=None
    df=None
    
    # Create GUI interface
    root = tk.Toplevel()
    root.title("Road Damage Detection")
    root.geometry("1340x700")
    root.resizable(False, False)
    root.configure(background='#EFE4B0')


    # Function to preprocess an image
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(224, 224))
        #img = load_img(image_path)
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 224.0  # Normalize pixel values to [0, 1]
        return img_array


    def create_object_detection_model(num_classes):
        #base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Changed model and input shape
        base_model = MobileNet(weights='mobilenet_1_0_224_tf_no_top.h5', include_top=False, input_shape=(224, 224, 3))  # Changed model and input shape

        for layer in base_model.layers:
            layer.trainable = False

        x = layers.GlobalAveragePooling2D()(base_model.output)
        # x = layers.Dense(512, activation='relu')(x)  # Consider removing this layer for smaller models
        x = layers.Dropout(0.5)(x)

        class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
        bbox_output = layers.Dense(4, name='bbox_output')(x)

        model = models.Model(inputs=base_model.input, outputs=[class_output, bbox_output])

        optimizer = Adam(learning_rate=0.0001)  # Adjust the learning rate
        model.compile(optimizer='Adam',
                      loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': Huber()},
                      metrics={'class_output': 'accuracy', 'bbox_output': 'mae'})
        '''
        model.compile(optimizer='adam',
                      loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': 'mse'},
                      metrics={'class_output': 'accuracy', 'bbox_output': 'mae'})
        '''

        return model


    print("Start Process")

    def Data_access():
        nonlocal df
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


        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig("1.png")
        
        frame = cv2.imread('1.png')
        frame = cv2.resize(frame, (800, 600))
        img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        camera_label.config(image=img)
        camera_label.image = img


    def Start_Preprocess():
        nonlocal images,class_labels,bbox_labels,history,df,X_train,X_test,y_train,y_test
        
        # Loop through DataFrame rows
        for _, row in df.iterrows():
            image_path = os.path.join(images_folder, row['filename'])
            #image = img_to_array(load_img(image_path, target_size=(224, 224)))
            #images.append(image)
            images.append(preprocess_image(image_path))

            label = row['class_id']

            # Normalize bounding box coordinates
            '''
            xmin = row['xmin'] / 224.0
            ymin = row['ymin'] / 224.0
            xmax = row['xmax'] / 224.0
            ymax = row['ymax'] / 224.0
            '''
            
            xmin = int((row['xmin']/row['width'])*224)
            xmax = int((row['xmax']/row['width'])*224)
            ymin = int((row['ymin']/row['height'])*224)
            ymax = int((row['ymax']/row['height'])*224)

            class_labels.append(label)
            bbox_labels.append([xmin, ymin, xmax, ymax])
            print([xmin, ymin, xmax, ymax])
            

        images = np.concatenate(images, axis=0)
        class_labels = np.array(class_labels)
        bbox_labels = np.array(bbox_labels)

        X_train, X_test, y_train, y_test = train_test_split(images, class_labels, test_size=0.2, random_state=42)

        print(len(images))
        print(len(class_labels))
        print(len(bbox_labels))


    def Start_TrainModel():
        nonlocal images,class_labels,bbox_labels,history
        print("start modeling")
        # Create and compile the model
        num_classes = len(set(class_labels))
        print(num_classes)
        model = create_object_detection_model(num_classes)

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
        plt.savefig("2.png")
        #plt.show()
        frame = cv2.imread('2.png')
        frame = cv2.resize(frame, (800, 600))
        img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        camera_label.config(image=img)
        camera_label.image = img


    def Start_TestModel():
        model = tf.keras.models.load_model('object_detection_model.h5')

        class_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
        print(f'Train Classification Accuracy: {class_accuracy * 100:.2f}%')

        class_accuracy1 = model.evaluate(X_test, y_test, verbose=0)[1]
        print(f'Test Classification Accuracy: {class_accuracy1 * 100:.2f}%')
        
    def plot_model_analysis():
        plot_training_history(history)

    label = tk.Label(root ,width=60,text = "Road Damage Detection",font=("arial italic", 30), bg="#0000FF", fg="white").grid(row=0, column=0,columnspan=2)

    button1 = tk.Button(root, text="Access Data", font=("Arial", 12),bg="#0000FF", fg="white", width=30,command=lambda:Data_access())
    button1.grid(row=1, column=0, padx=10, pady=5)

    button2 = tk.Button(root, text="Image Pre-Processing", font=("Arial", 12),bg="#0000FF", fg="white", width=30,command=lambda:Start_Preprocess())
    button2.grid(row=2, column=0, padx=10, pady=5)

    button3 = tk.Button(root, text="Train Model", font=("Arial", 12),bg="#0000FF", fg="white", width=30,command=lambda:Start_TrainModel())
    button3.grid(row=3, column=0, padx=10, pady=5)

    button5 = tk.Button(root, text="Test Model", font=("Arial", 12),bg="#0000FF", fg="white", width=30,command=lambda:Start_TestModel())
    button5.grid(row=4, column=0, padx=10, pady=5)
    
    button4 = tk.Button(root, text="Model Analysis", font=("Arial", 12),bg="#0000FF", fg="white", width=30,command=lambda: plot_model_analysis())
    button4.grid(row=5, column=0, padx=10, pady=5)
 
    # Display live camera feed
    camera_label = tk.Label(root,width=800, height=600, borderwidth=2, relief="solid")
    camera_label.grid(row=1, column=1, rowspan=5, padx=10, pady=10)
    frame = cv2.imread('No.png')
    frame = cv2.resize(frame, (800, 600))
    img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
    camera_label.config(image=img)
    camera_label.image = img
        
    # Start the GUI loop
    root.mainloop()

