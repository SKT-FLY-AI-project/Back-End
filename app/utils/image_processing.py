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
# def detect_painting_region(image, min_area_ratio=0.2, aspect_ratio_range=(0.75, 1.5)):      
#     # aspect_ratio_range ✅ 가로/세로 비율 고려 → 너무 넓거나 좁은 탐지를 방지
#     """
#     그림이 너무 작은 영역으로 잘려 어두워지는 문제를 방지하기 위해,
#     최소 면적 비율을 설정하여 너무 작은 영역은 무시하고 원본을 유지하도록 한다.
#     """
    
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     edges = cv2.Canny(gray, 100, 200)
#     edge = cv2.bitwise_or(thresh, edges)

#     # Contour Detection 적용
#     contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     detected_regions = []
    
#     for cnt in contours:
#         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#         if len(approx) == 4:
#             detected_regions.append(approx)
    
#     if detected_regions:
#         # 가장 큰 영역 선택
#         largest_region = max(detected_regions, key=cv2.contourArea)
#         ordered_points = order_points(largest_region.reshape(4, 2))
                
#         # 사각형 너비와 높이 계산
#         (tl, tr, br, bl) = ordered_points
#         width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
#         height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        
#         # 전체 이미지 대비 크롭된 영역이 너무 작으면 원본 반환
#         img_area = image.shape[0] * image.shape[1]
#         cropped_area = width * height
#         if cropped_area < min_area_ratio * img_area:
#             print("⚠️ 그림 영역이 너무 작아 원본 이미지를 반환합니다.")
#             return image
        
#         # 변환할 대상 좌표 설정
#         dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        
#         # 원근 변환 수행
#         M = cv2.getPerspectiveTransform(ordered_points, dst)
#         corrected = cv2.warpPerspective(image, M, (width, height))
#         return corrected
    
#     print("⚠️ 사각형을 찾을 수 없습니다. 원본 이미지를 반환합니다.")
#     return image

def detect_painting_region(image):
    """
    그림의 영역을 자동 감지하여 정확하게 크롭하는 함수
    - Bounding Box 대신 minAreaRect() 활용
    - Perspective Transform을 항상 적용하여 그림 외부 배경 제거
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1️⃣ 노이즈 제거를 위한 Gaussian Blur 추가 적용 (벽 패턴 제거 강화)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.bilateralFilter(blurred, 9, 75, 75)

    # 2️⃣ Adaptive Thresholding 적용 (내부 디테일 줄이고 경계 강조)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # 3️⃣ Canny Edge Detection 적용 (그림 테두리만 강조)
    edges = cv2.Canny(gray, 50, 150)
    edge_combined = cv2.bitwise_or(adaptive_thresh, edges)

    # 4️⃣ Morphological Closing 적용 (작은 노이즈 제거 및 경계 강화)
    kernel = np.ones((5,5), np.uint8)
    edge_processed = cv2.morphologyEx(edge_combined, cv2.MORPH_CLOSE, kernel)

    # 5️⃣ Contour Detection 적용
    contours, _ = cv2.findContours(edge_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠️ 윤곽선을 찾을 수 없습니다. 원본을 반환합니다.")
        return image

    # 6️⃣ 가장 큰 Contour 선택 (그림을 포함하는 영역)
    largest_contour = max(contours, key=cv2.contourArea)

    # 7️⃣ 최소 회전 사각형 (minAreaRect) 적용하여 그림 크기 감지
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    # 8️⃣ 좌표 정렬
    ordered_points = order_points(box)

    # 9️⃣ Perspective Transform 수행하여 그림만 정확하게 크롭
    width = int(max(np.linalg.norm(ordered_points[0] - ordered_points[1]),
                    np.linalg.norm(ordered_points[2] - ordered_points[3])))
    height = int(max(np.linalg.norm(ordered_points[0] - ordered_points[3]),
                     np.linalg.norm(ordered_points[1] - ordered_points[2])))

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_points, dst)
    corrected = cv2.warpPerspective(image, M, (width, height))

    return corrected

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