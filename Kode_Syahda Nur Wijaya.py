import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2

# Baca dataset colors.csv
color_data = pd.read_csv('datasheet/colors.csv')
X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Akurasi model KNN pada data test: {accuracy_score(y_test, y_pred)*100:.2f}%")

color_bgr_map = {
    "Red": (0, 0, 255), "Yellow": (0, 255, 255), "Green": (0, 255, 0),
    "Blue": (255, 0, 0), "Cyan": (255, 255, 0), "Magenta": (255, 0, 255),
    "Black": (0, 0, 0), "White": (255, 255, 255)
}

def detect_color_hsv(hsv_pixel):
    # rentang HSV warna
    lower_red1 = np.array([0,120,70]); upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,120,70]); upper_red2 = np.array([179,255,255])
    lower_yellow = np.array([20,100,100]); upper_yellow = np.array([30,255,255])
    lower_green = np.array([40,50,50]); upper_green = np.array([80,255,255])
    lower_blue = np.array([90,50,50]); upper_blue = np.array([130,255,255])

    mask_red = cv2.inRange(hsv_pixel, lower_red1, upper_red1) + cv2.inRange(hsv_pixel, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv_pixel, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_pixel, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv_pixel, lower_blue, upper_blue)

    if mask_red[0] > 0: return "Red"
    elif mask_yellow[0] > 0: return "Yellow"
    elif mask_green[0] > 0: return "Green"
    elif mask_blue[0] > 0: return "Blue"
    else: return None

def draw_label_and_box(image, text, top_left, bottom_right, color, acc_text=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2; font_scale = 1.2

    # Gambar bounding box
    cv2.rectangle(image, top_left, bottom_right, color, 3)
    # Background transparan
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

    brightness = (color[0]*0.299 + color[1]*0.587 + color[2]*0.114)
    text_color = (255,255,255) if brightness<128 else (0,0,0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = top_left[0] + (bottom_right[0]-top_left[0]-text_size[0])//2
    text_y = top_left[1] + (bottom_right[1]-top_left[1]+text_size[1])//2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)

    if acc_text:
        cv2.putText(image, acc_text, (10,30), font, 0.8, (0,255,0), 2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error membuka kamera")
    exit()

total_frames = 0
correct_predictions = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width, _ = frame.shape
    center_x, center_y = width//2, height//2
    roi_w, roi_h = 120, 120
    top_left = (center_x - roi_w//2, center_y - roi_h//2)
    bottom_right = (center_x + roi_w//2, center_y + roi_h//2)

    pixel_bgr = frame[center_y, center_x].reshape(1,1,3)
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
    color_special = detect_color_hsv(pixel_hsv)

    if color_special:
        color_pred = color_special
    else:
        pixel_rgb = pixel_bgr[0][0][::-1].reshape(1,-1)
        pixel_rgb_scaled = scaler.transform(pixel_rgb)
        color_pred = knn.predict(pixel_rgb_scaled)[0]

    total_frames += 1
    if color_special == color_pred or color_special is not None:
        correct_predictions += 1
    accuracy_percent = (correct_predictions / total_frames)*100

    box_color = color_bgr_map.get(color_pred, (0,0,0))
    acc_text = f"Akurasi Deteksi Warna: {accuracy_percent:.2f}%"
    draw_label_and_box(frame, color_pred, top_left, bottom_right, box_color, acc_text)

    cv2.imshow('Deteksi Warna dengan KNN dan Akurasi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()