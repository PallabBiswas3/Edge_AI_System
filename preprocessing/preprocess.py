import cv2
import numpy as np

def simulate_edge_noise(image):
    noise = np.random.normal(0, 5, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def reduce_resolution(image):
    small = cv2.resize(image, (16,16))
    return cv2.resize(small, (32,32))
