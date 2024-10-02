
from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL.Image

class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):
        img_list = np.array([]).reshape(0, 22500)  # Initialize with the correct shape
        class_list = np.array([])

        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)  # Load image in grayscale
            img = cv.resize(img, (150, 150))  # Ensure image is resized to 150x150
            img = img.reshape(22500)  # Reshape the image to 1D array of size 22500
            img_list = np.vstack([img_list, img])  # Add the image to the array
            class_list = np.append(class_list, 1)

        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)  # Load image in grayscale
            img = cv.resize(img, (150, 150))  # Ensure image is resized to 150x150
            img = img.reshape(22500)  # Reshape the image to 1D array of size 22500
            img_list = np.vstack([img_list, img])  # Add the image to the array
            class_list = np.append(class_list, 2)

        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]  # Extract the frame data
        gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)  # Convert to grayscale
        resized_frame = cv.resize(gray_frame, (150, 150))  # Resize to 150x150
        reshaped_frame = resized_frame.reshape(22500)  # Flatten the image to match the trained model

        prediction = self.model.predict([reshaped_frame])

        return prediction[0]

