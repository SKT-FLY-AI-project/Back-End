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

    # Otsu's thresholding 적용
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive Thresholding 적용 (Otsu 대비 보완)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Canny Edge Detection의 동적 임계값 설정
    median_val = np.median(gray)
    lower = int(max(0, 0.33 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(gray, lower, upper)

    # Thresholding과 Edge Detection 결합
    combined_edges = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
    combined_edges = cv2.bitwise_or(combined_edges, edges)

    # Morphological Closing 적용하여 경계 보정
    kernel = np.ones((3,3), np.uint8)
    processed_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

    return processed_edges

# 주요 색상 추출
def extract_dominant_colors(image, k=5):
    image = image.reshape((-1, 3))
    image = np.float32(image)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    dominant_colors = palette[np.argsort(-counts)]
    return dominant_colors.astype(int)

def adjust_hsv_lightness_and_saturation(rgb, lightness_factor=1.4, saturation_factor=1.3):
    """
    HSV 색 공간에서 명도(Value)와 채도(Saturation)을 조정하여  
    사람이 인식하는 색감과 비슷하게 변환하는 함수.

    - `lightness_factor`: 명도(Value) 조정 강도
    - `saturation_factor`: 채도(Saturation) 조정 강도
    """
    # RGB → HSV 변환
    rgb_array = np.array([[rgb]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[0, 0]  # 단일 픽셀 값 추출

    # ✅ 명도(Value) 조정
    if v < 150:  # 기존보다 어두운 색상은 더 밝게
        v = min(v * lightness_factor, 255)
    elif v > 220:  # 너무 밝은 색상은 과하지 않게 보정
        v = min(v * 1.1, 255)

    # ✅ 채도(Saturation) 증가하여 원색 계열을 더 살림
    s = min(s * saturation_factor, 255)

    # ✅ 특정 색 계열(파란색, 노란색, 초록색 등)에 대한 추가 보정
    if 180 <= h <= 260:  # 파란색 계열
        v = min(v * 1.4, 255)
        s = min(s * 1.3, 255)
    elif 40 <= h <= 80:  # 노란색 계열
        v = min(v * 1.5, 255)
        s = min(s * 1.4, 255)
    elif 80 <= h <= 160:  # 초록색 계열
        v = min(v * 1.4, 255)
        s = min(s * 1.3, 255)

    # HSV → RGB 변환
    new_hsv = np.array([[[h, int(s), int(v)]]], dtype=np.uint8)
    new_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)[0, 0]

    return tuple(new_rgb)

def get_color_name(rgb):
    """
    RGB 값을 HSV 기반으로 사람이 인식하기 쉬운 색상 계열로 변환하는 함수.
    푸른색 인식률 개선 버전.
    """

    # ✅ RGB → HSV 변환
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv

    # ✅ 색상 카테고리 정의 (HSV Hue 기준, 갈색을 넓히고 중복 해결)
    color_categories = {
        "푸른색": (85, 165),  # 푸른색 범위 확장 (기존: 90, 150)
        "하늘색": (150, 190),  # 하늘색 범위 확장 (기존: 150, 180)
        "민트색": (170, 195),
        "초록색": (45, 90),   
        "노란색": (20, 45),   
        "주황색": (10, 20),   
        "붉은색": [(0, 10), (330, 360)],  
        "보라색": (265, 330), # 보라색 범위 조정 (기존: 270, 330)
        "갈색": (0, 20),      
    }

    result_colors = set()  # 다중 색상 결과를 담을 리스트 (중복 제거)

    # HSV의 H값은 0-179 범위인 경우가 있으므로 정규화
    # OpenCV의 H는 0-179, S와 V는 0-255 범위
    h_normalized = h * 2 if h <= 90 else h  # H값 정규화 (OpenCV에서는 0-180)
    
    # ✅ 푸른색 계열 우선 검사 (회색 판단 전)
    is_blue_range = False
    if (85 <= h_normalized < 165) or (180 <= h_normalized <= 260):
        is_blue_range = True
        
    # ✅ 회색 및 밝기 계열 분류 (채도가 낮을 때)
    # 푸른색 계열은 더 낮은 채도(30)에서도 푸른색으로 인식
    if is_blue_range and s < 30:
        if v > 150:  # 명도가 높으면 하늘색
            result_colors.add("하늘색")
        else:
            result_colors.add("푸른색")
    elif s < 40:  # 다른 색상들의 채도 기준
        if is_blue_range:  # 푸른색 범위면 푸른색 유지
            if v > 180:
                result_colors.add("하늘색")
            else:
                result_colors.add("푸른색")
        elif 80 <= h_normalized <= 160:  # 초록색 계열이면 초록색 유지
            result_colors.add("초록색")
        else:
            if v < 80:
                result_colors.add("어두운 색")
            elif v > 200:
                result_colors.add("밝은 색")
            else:
                result_colors.add("회색")
    else:
        # 색상 범주 매칭 로직
        for category, hue_range in color_categories.items():
            if isinstance(hue_range, tuple):
                if hue_range[0] <= h_normalized < hue_range[1]:
                    result_colors.add(category)
            elif isinstance(hue_range, list):
                for hr in hue_range:
                    if hr[0] <= h_normalized < hr[1]:
                        result_colors.add(category)

    # 노란색과 갈색의 구분
    if 0 <= h <= 20 and s < 100 and v < 150:  # 낮은 채도, 낮은 명도일 때만 갈색
        result_colors.add("갈색")
    elif 20 <= h <= 45:  # 노란색은 범위 유지
        result_colors.add("노란색")

    # 채도와 명도에 따른 추가 분류
    if s < 100 and v < 100 and not is_blue_range:  # 푸른색 범위가 아닐 때만 갈색 추가
        result_colors.add("갈색")
        
    # 푸른색과 하늘색 보정 - 명도에 따른 구분
    if "푸른색" in result_colors or "하늘색" in result_colors:
        if v > 180:  # 명도가 높으면 하늘색
            if "하늘색" not in result_colors:
                result_colors.add("하늘색")
            if "푸른색" in result_colors and h_normalized < 180:
                result_colors.remove("푸른색")
        elif v < 120 and s > 50:  # 명도가 낮고 채도가 충분하면 진한 푸른색
            if "하늘색" in result_colors:
                result_colors.remove("하늘색")
            if "푸른색" not in result_colors:
                result_colors.add("푸른색")

    # 최종 색상 리스트 정리 (중복 제거 + 정렬)
    result_colors = sorted(result_colors)  # 정렬하여 일관된 순서 유지

    # 최종 색상 리스트 반환
    return ", ".join(result_colors) if result_colors else "회색"

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