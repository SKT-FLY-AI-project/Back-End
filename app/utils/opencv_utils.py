import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 이미지 로드 및 전처리
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상단
    rect[2] = pts[np.argmax(s)]  # 우하단
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상단
    rect[3] = pts[np.argmax(diff)]  # 좌하단
    return rect

# # 그림 영역 탐지 (Contour Detection)
def detect_painting_region(image, min_area_ratio=0.2, aspect_ratio_range=(0.75, 1.5)):      
    # aspect_ratio_range ✅ 가로/세로 비율 고려 → 너무 넓거나 좁은 탐지를 방지
    """
    그림이 너무 작은 영역으로 잘려 어두워지는 문제를 방지하기 위해,
    최소 면적 비율을 설정하여 너무 작은 영역은 무시하고 원본을 유지하도록 한다.
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(gray, 100, 200)
    edge = cv2.bitwise_or(thresh, edges)

    # Contour Detection 적용
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_regions = []
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            detected_regions.append(approx)
    
    if detected_regions:
        # 가장 큰 영역 선택
        largest_region = max(detected_regions, key=cv2.contourArea)
        ordered_points = order_points(largest_region.reshape(4, 2))
                
        # 사각형 너비와 높이 계산
        (tl, tr, br, bl) = ordered_points
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        
        # 전체 이미지 대비 크롭된 영역이 너무 작으면 원본 반환
        img_area = image.shape[0] * image.shape[1]
        cropped_area = width * height
        if cropped_area < min_area_ratio * img_area:
            print("⚠️ 그림 영역이 너무 작아 원본 이미지를 반환합니다.")
            return image
        
        # 변환할 대상 좌표 설정
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        
        # 원근 변환 수행
        M = cv2.getPerspectiveTransform(ordered_points, dst)
        corrected = cv2.warpPerspective(image, M, (width, height))
        return corrected
    
    print("⚠️ 사각형을 찾을 수 없습니다. 원본 이미지를 반환합니다.")
    return image


# 주요 객체 검출 (Edge Detection 사용)
def detect_edges(image):
    image = detect_painting_region(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Adaptive Thresholding 적용하여 대비 강화 
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Canny Edge Detection 적용
    edges = cv2.Canny(gray, 100, 200)
    # Thresholding과 Edge Detection 결합
    combined_edges = cv2.bitwise_or(thresh, edges)
    return combined_edges

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