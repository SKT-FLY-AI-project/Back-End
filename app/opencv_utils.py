########################### SETP 1 : openCV #####################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 이미지 로드 및 전처리
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# # 그림 영역 탐지 (Contour Detection)
def detect_painting_region(image, min_area_ratio=0.2):
    """
    그림이 너무 작은 영역으로 잘려 어두워지는 문제를 방지하기 위해,
    최소 면적 비율을 설정하여 너무 작은 영역은 무시하고 원본을 유지하도록 한다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_regions = [
        cv2.boundingRect(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) 
        for cnt in contours 
        if len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4
    ]
    
    if detected_regions:
        # 가장 큰 사각형 영역 선택
        x, y, w, h = max(detected_regions, key=lambda r: r[2] * r[3])
        
        # 전체 이미지 대비 크롭된 영역이 너무 작으면 원본 반환
        img_area = image.shape[0] * image.shape[1]
        cropped_area = w * h
        if cropped_area < min_area_ratio * img_area:
            print("⚠️ 그림 영역이 너무 작아 원본 이미지를 반환합니다.")
            return image
        
        return image[y:y+h, x:x+w]
    
    return image


# 주요 객체 검출 (Edge Detection 사용)
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 주요 색상 추출
def extract_dominant_colors(image, k=5):
    image = image.reshape((-1, 3))
    image = np.float32(image)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    dominant_colors = palette[np.argsort(-counts)]
    return dominant_colors.astype(int)

def get_color_name(rgb):
    """ RGB 값을 가장 가까운 색상명으로 변환 """
    min_dist = float('inf')
    closest_color = "알 수 없는 색"
    
    for name, hex in mcolors.CSS4_COLORS.items():
        r, g, b = mcolors.hex2color(hex)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        dist = np.sqrt((r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2)
        
        if dist < min_dist:
            min_dist = dist
            closest_color = name

    return closest_color

# 결과 시각화
def display_results(image_path):
    image = load_and_preprocess_image(image_path)
    painting_region = detect_painting_region(image)  # 밝기 조정 없이 원본 그대로 사용
    edges = detect_edges(image)
    dominant_colors = extract_dominant_colors(painting_region)
    
    
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    plt.subplot(1, 4, 2)
    plt.imshow(painting_region)
    plt.title("Detected Painting Region")

    plt.subplot(1, 4, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    
    plt.subplot(1, 4, 4)
    plt.imshow([dominant_colors / 255])
    plt.title("Dominant Colors (Original)")

    plt.show()
    return edges, dominant_colors