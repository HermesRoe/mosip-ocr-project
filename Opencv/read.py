import cv2 as cv
import numpy as np

# =========================================================
# SETTINGS
# =========================================================
BLUR_THRESHOLD = 100  # Adjust experimentally (~100 for typical images)

# =========================================================
# FUNCTION: Resize any image to fit inside a 1920x1080 window
# =========================================================
def fit_to_window(img, max_w=1920, max_h=1080):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized

# =========================================================
# FUNCTION: Blur Detection (Pre-OCR Quality Check)
# =========================================================
def detect_blur(image, threshold=BLUR_THRESHOLD):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    fm = cv.Laplacian(gray, cv.CV_64F).var()
    is_blurry = fm < threshold

    print("\n----- BLUR DETECTION (Pre-OCR) -----")
    print(f"Laplacian Variance: {fm}")
    if is_blurry:
        print(f"Result: Image is BLURRY (Threshold={threshold}) → FAILED THIS QUALITY CHECK")
    else:
        print(f"Result: Image is SHARP (Threshold={threshold}) → PASSED THIS QUALITY CHECK")

    return fm, is_blurry

# =========================================================
# FUNCTION: Ghost Image Detection (Pre-OCR Quality Check)
# =========================================================
def detect_ghost(image, diff_threshold=50):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    edges = cv.magnitude(sobelx, sobely)

    shifted = np.roll(edges, 5, axis=1)
    diff = cv.absdiff(edges.astype(np.uint8), shifted.astype(np.uint8))
    avg_diff = np.mean(diff)
    is_ghosting = avg_diff > diff_threshold

    print("\n----- GHOST IMAGE DETECTION (Pre-OCR) -----")
    print(f"Edge Difference Score: {avg_diff}")
    if is_ghosting:
        print(f"Result: POSSIBLE GHOSTING detected (Threshold={diff_threshold}) → FAILED THIS QUALITY CHECK")
    else:
        print(f"Result: No ghosting detected (Threshold={diff_threshold}) → PASSED THIS QUALITY CHECK")

    return avg_diff, is_ghosting

# =========================================================
# MAIN PROGRAM
# =========================================================
image_path = "Images/catz.jpg"

# Load original image
img = cv.imread(image_path)
if img is None:
    print("Error: Image not found")
    exit()

# =========================================================
# PRE-OCR QUALITY CHECK on ORIGINAL IMAGE
# =========================================================
fm, is_blurry = detect_blur(img, threshold=BLUR_THRESHOLD)
ghost_score, is_ghosting = detect_ghost(img, diff_threshold=50)

# =========================================================
# FINAL QUALITY CHECK
# =========================================================
if is_blurry or is_ghosting:
    print("\nFINAL RESULT: IMAGE FAILED ALL QUALITY CHECKS → Do NOT run OCR")
else:
    print("\nFINAL RESULT: IMAGE PASSED ALL QUALITY CHECKS → Ready for OCR")

# =========================================================
# DISPLAY IMAGES
# =========================================================
# Display ORIGINAL image
cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
cv.imshow("Original Image", img)

# Resize copy for display
resized_img = fit_to_window(img, 1920, 1080)
cv.namedWindow("Resized Image for Display", cv.WINDOW_NORMAL)
cv.imshow("Resized Image for Display", resized_img)

cv.waitKey(0)
cv.destroyAllWindows()

# =========================================================
# OPTIONAL: Save resized copy
# =========================================================
cv.imwrite("resized_output.jpg", resized_img)

# =========================================================
# OCR Function Placeholder
# =========================================================
def extract_text(image):
    # Only run if image passes quality check
    print("\n[INFO] Extracting text... (OCR not implemented in this snippet)")
