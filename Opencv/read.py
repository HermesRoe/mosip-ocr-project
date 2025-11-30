import cv2 as cv
import numpy as np
import math
from paddleocr import PaddleOCR
from difflib import get_close_matches
import json

# ============================================================
# FIT-TO-WINDOW FOR DISPLAY
# ============================================================
def fit_to_window(img, max_w=1920, max_h=1080):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

# ============================================================
# BLUR DETECTION USING LAPLACIAN VARIANCE
# ============================================================
def detect_blur_laplacian(img_gray, threshold=100):
    lap = cv.Laplacian(img_gray, cv.CV_64F)
    variance = lap.var()
    return variance, variance < threshold

# ============================================================
# EDGE SHARPNESS SCORE
# ============================================================
def edge_sharpness_score(img_gray):
    edges = cv.Canny(img_gray, 100, 200)
    edge_score = np.sum(edges == 255) / (img_gray.shape[0] * img_gray.shape[1])
    return edge_score

# ============================================================
# IMPROVED GHOST DETECTION
# ============================================================
def detect_ghost_improved(img_color):

    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=4, minSize=(20,20))
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    face_count = len(faces)

    has_ghost = False
    ghost_confidence = 0.0
    ghost_rect = None
    main_rect = None

    if face_count >= 2:
        x1,y1,w1,h1 = faces[0]
        x2,y2,w2,h2 = faces[1]
        main_rect = (x1,y1,w1,h1)
        ghost_rect = (x2,y2,w2,h2)
        main_area = w1*h1
        ghost_area = w2*h2
        size_ratio = ghost_area / main_area
        ghost_crop = img_gray[y2:y2+h2, x2:x2+w2]
        ghost_contrast = np.std(ghost_crop)/255.0
        cx1,cy1 = x1+w1/2, y1+h1/2
        cx2,cy2 = x2+w2/2, y2+h2/2
        dist = math.dist((cx1,cy1), (cx2,cy2))
        norm_dist = dist / max(1.0, w1)
        size_ok = 0.08<=size_ratio<=0.6
        near_ok = norm_dist<=1.8
        if size_ok and near_ok:
            ghost_confidence = float(np.clip(round((size_ratio*(1-ghost_contrast))/0.5,3),0.0,1.0))
        has_ghost = ghost_confidence>0.2
        return has_ghost, face_count, ghost_confidence, main_rect, ghost_rect

    if face_count == 1:
        main_rect = tuple(faces[0])
    else:
        main_rect = None

    return False, face_count, 0.0, main_rect, ghost_rect

# ============================================================
# OCR EXTRACTION WITH SAFE TYPES
# ============================================================
def extract_text_paddleocr(image_path):
    img = cv.imread(image_path)
    if img is None:
        return [], {"Name":"", "DOB":"", "ID":"", "Address":""}

    # Resize for processing (fit to 1920x1080)
    img_proc = fit_to_window(img, 1920, 1080)

    gray = cv.cvtColor(img_proc, cv.COLOR_BGR2GRAY)
    # Sharpen & contrast enhance
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv.filter2D(gray, -1, kernel)
    gray = cv.convertScaleAbs(gray, alpha=1.5, beta=0)
    prep_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(prep_img)

    parsed = []
    fields = {"Name": "", "DOB": "", "ID": "", "Address": ""}

    for line in result:
        for word in line:
            if len(word)<2:
                continue
            box = word[0]
            info = word[1]
            if isinstance(info,(tuple,list)) and len(info)==2:
                text = str(info[0])
                conf = float(info[1])
            else:
                text = str(info)
                conf = 0.0
            # Ensure box coordinates are numeric
            safe_box = []
            for pt in box:
                try:
                    safe_box.append([int(float(pt[0])), int(float(pt[1]))])
                except:
                    safe_box.append([0,0])
            parsed.append({"text":text,"confidence":conf,"box":safe_box})

    # Fuzzy match fields
    labels = {"Name":["name"], "DOB":["dob","date of birth"], "ID":["id","identification"], "Address":["address"]}
    texts = [p["text"] for p in parsed]
    for i, text in enumerate(texts):
        t_lower = text.lower()
        for key, variants in labels.items():
            for label in variants:
                if get_close_matches(label,[t_lower],cutoff=0.6):
                    # take next few words
                    value_words = texts[i+1:i+4] if i+1<len(texts) else []
                    value = " ".join(value_words).strip()
                    if value:
                        fields[key] = value
                    else:
                        if ":" in text:
                            fields[key] = text.split(":",1)[1].strip()
                    break

    return parsed, fields, img_proc

# ============================================================
# DRAW BOXES WITH CONFIDENCE COLORS
# ============================================================
def draw_boxes(img, parsed):
    overlay = img.copy()
    for item in parsed:
        try:
            box = np.array([[int(float(x)), int(float(y))] for x, y in item["box"]], dtype=int)
        except:
            continue
        conf = float(item.get("confidence",0))
        if conf>0.8:
            color=(0,255,0) # high
        elif conf>0.5:
            color=(0,255,255) # medium
        else:
            color=(0,0,255) # low
        cv.polylines(overlay,[box],isClosed=True,color=color,thickness=2)
        # put text
        cv.putText(overlay, item["text"], tuple(box[0]), cv.FONT_HERSHEY_SIMPLEX, 0.6, color,2)
    return overlay

# ============================================================
# SAFE JSON DUMP
# ============================================================
def safe_json_output(data):
    def convert(obj):
        if isinstance(obj,(np.float32,np.float64)):
            return float(obj)
        if isinstance(obj,(np.int32,np.int64)):
            return int(obj)
        if isinstance(obj,(np.bool_)):
            return bool(obj)
        return obj
    return json.dumps(data, default=convert, indent=4)

# ============================================================
# MAIN
# ============================================================
image_path = "Images/girl1.jpg"
img = cv.imread(image_path)
if img is None:
    print("Error: image not found")
    exit()

# Blur/sharpness
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
lap_var, is_blur = detect_blur_laplacian(gray)
edge_score = edge_sharpness_score(gray)

# Ghost detection
has_ghost, face_count, ghost_conf, main_rect, ghost_rect = detect_ghost_improved(img)

# OCR extraction
ocr_list, field_dict, img_proc = extract_text_paddleocr(image_path)

# Draw boxes
annotated_img = draw_boxes(img_proc, ocr_list)
cv.imshow("OCR & Bounding Boxes", annotated_img)
cv.waitKey(0)
cv.destroyAllWindows()

# Prepare JSON
output = {
    "quality_check": {
        "is_blurry": bool(is_blur),
        "blur_score": float(lap_var),
        "ghost_image_detected": bool(has_ghost),
        "ghost_confidence": float(ghost_conf),
        "face_count": int(face_count)
    },
    "ocr_data": ocr_list,
    "fields": field_dict,
    "status": "success"
}

print(safe_json_output(output))
