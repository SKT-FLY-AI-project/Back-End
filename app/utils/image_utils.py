from PIL import Image
import numpy as np
import cv2

def preprocess_image(image: Image):
    image = image.convert("L")  # Convert image to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_np = np.array(image).astype("float32") / 255.0
    image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def detect_painting_region(image):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_regions = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            detected_regions.append(cv2.boundingRect(approx))
    
    if detected_regions:
        x, y, w, h = max(detected_regions, key=lambda r: r[2] * r[3])
        return Image.fromarray(image_np[y:y+h, x:x+w])
    else:
        return image
