import tkinter as tk
import cv2
import time
import threading
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('object_detection_model.h5')

def Detect_Start(URL):
    
    # Create GUI interface
    root = tk.Toplevel()
    root.title("Road Damage Detection")
    root.geometry("840x650")
    root.resizable(False, False)
    root.configure(background='#EFE4B0')
    
    label = tk.Label(root ,width=40,text = "Road Damage Detection",font=("arial italic", 30), bg="#0000FF", fg="white").grid(row=0, column=0,columnspan=2)

    # Display live camera feed
    camera_label = tk.Label(root,width=600, height=500, borderwidth=2, relief="solid")
    camera_label.grid(row=1, column=0,padx=10, pady=10)

    Cval = tk.StringVar()
    label = tk.Label(root ,width=40,font=("arial italic", 20), bg="#0000FF", fg="white",textvariable=Cval).grid(row=2, column=0)
    Cval.set("Detected Yes/No")
    '''
    # Function to preprocess an image
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(224, 224))
        #img = load_img(image_path)
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        return img_array
    '''

    def preprocess_image(image):
        # Resize the image to the model's input size
        img_array = cv2.resize(image, (224, 224))
        img_array = img_to_array(img_array)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        return img_array

    # Start live camera feed
    def show_camera():
        frame = cv2.imread(URL)
        frame = cv2.resize(frame, (224, 224))
        new_image = preprocess_image(frame)
        # Make predictions
        class_probs, bbox_preds = loaded_model.predict(new_image)
        # Get the predicted class
        predicted_class = np.argmax(class_probs)
        print(predicted_class)
        # Get the predicted bounding box coordinates (unnormalized)
        predicted_bbox = bbox_preds[0]
        print(predicted_bbox)
        # Get the bounding box coordinates
        x, y, width, height = predicted_bbox
        #x=x*224.0
        #y=y*224.0
        #width=width*224.0
        #height=height*224.0
        
        width = width - x
        height = height - y

        
        cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)
        img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        camera_label.config(image=img)
        camera_label.image = img

        
    show_camera()
    # Create a separate thread for the camera feed
    print("Start")

    
    # Start the GUI loop
    root.mainloop()

