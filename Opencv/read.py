import os, sys, re, json, math
from difflib import get_close_matches
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

# try TrOCR (HuggingFace)
USE_TROCR = False
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model chosen is mixed-handwritten; it works well for mixed printed/handwritten text.
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)
    USE_TROCR = True
    print("[INFO] TrOCR loaded and will be used for recognition (device=%s)." % device)
except Exception as e:
    print("[WARN] TrOCR not available or failed to load. Falling back to PaddleOCR for recognition. Error:", e)
    USE_TROCR = False

# try CRAFT (detection). If not installed, we'll fallback to MSER detection
USE_CRAFT = False
try:
    from craft_text_detector import Craft
    craft = Craft(output_dir=None, export_extra=False, refine_net=True)  # no file outputs
    USE_CRAFT = True
    print("[INFO] CRAFT text detector loaded.")
except Exception as e:
    print("[WARN] CRAFT not available - falling back to MSER detection. Error:", e)
    USE_CRAFT = False

# optional fallback recognizer: PaddleOCR (if TrOCR missing)
USE_PADDLE_FALLBACK = False
try:
    if not USE_TROCR:
        from paddleocr import PaddleOCR
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        USE_PADDLE_FALLBACK = True
        print("[INFO] PaddleOCR loaded as fallback recognizer.")
except Exception:
    USE_PADDLE_FALLBACK = False

# --- Field configuration & regexes ---
TARGET_FIELDS = ["Name", "DOB", "Gender", "BloodGroup", "Phone", "Email", "Address", "IDNumber", "PinCode"]
LABEL_VARIANTS = {
    "Name": ["name", "full name", "first name"],
    "DOB": ["dob", "date of birth", "birth", "d.o.b", "date of birth:"],
    "Gender": ["gender", "sex"],
    "BloodGroup": ["blood", "blood group", "bloodtype"],
    "Phone": ["phone", "phone number", "mobile", "contact", "tel"],
    "Email": ["email", "e-mail", "email id"],
    "Address": ["address", "addr", "address line"],
    "IDNumber": ["id", "id no", "id number", "aadhar", "pan", "card no"],
    "PinCode": ["pin", "pincode", "postal", "zip"]
}
RE_EMAIL = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', flags=re.I)
RE_PHONE = re.compile(r'(\+?\d[\d\-\s]{6,}\d)')
RE_DOB = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
RE_AADHAAR = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
RE_PAN = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b', flags=re.I)
RE_BLOOD = re.compile(r'\b(?:A|B|AB|O)[+-]\b', flags=re.I)
RE_PIN = re.compile(r'\b\d{5,7}\b')

# --- Helpers ---
def safe_json(obj):
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
    return json.dumps(obj, default=convert, indent=2, ensure_ascii=False)

def normalize_box(box):
    if not box:
        return [[0,0],[0,0],[0,0],[0,0]]
    pts = []
    for pt in box:
        try:
            x = int(round(float(pt[0]))); y = int(round(float(pt[1])))
        except Exception:
            x, y = 0, 0
        pts.append([x,y])
    if len(pts) == 2:
        (x0,y0),(x1,y1) = pts
        return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    if len(pts) < 4:
        while len(pts) < 4:
            pts.append(pts[-1] if pts else [0,0])
    return pts[:4]

def iou_box(a,b):
    try:
        ax1 = min(pt[0] for pt in a); ay1 = min(pt[1] for pt in a)
        ax2 = max(pt[0] for pt in a); ay2 = max(pt[1] for pt in a)
        bx1 = min(pt[0] for pt in b); by1 = min(pt[1] for pt in b)
        bx2 = max(pt[0] for pt in b); by2 = max(pt[1] for pt in b)
    except Exception:
        return 0.0
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    union = area_a+area_b-inter
    return inter/union if union>0 else 0.0

# ---------------- Quality checks ----------------
def detect_blur_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    f = np.fft.fft2(gray); fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_mean = float(np.mean(mag))
    is_blur = (lap_var < 150.0) and (mag_mean < 30000.0)
    return {"is_blurry": bool(is_blur), "lap_var": lap_var, "fft_mean": mag_mean}

def detect_ghost_via_template(img):
    # rough duplicate-region detection using edges + template matching
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 400: continue
        rects.append((x,y,w,h,w*h))
    rects = sorted(rects, key=lambda r: r[4], reverse=True)
    ghost_conf = 0.0; ghost_rect=None; main_rect=None
    for i in range(min(4, len(rects))):
        x1,y1,w1,h1,_ = rects[i]; main_rect=(int(x1),int(y1),int(w1),int(h1))
        crop1 = gray[y1:y1+h1, x1:x1+w1]
        for j in range(i+1, min(len(rects), i+6)):
            x2,y2,w2,h2,_ = rects[j]
            ratio = float(w2*h2)/max(1.0,w1*h1)
            if ratio < 0.02 or ratio > 0.6: continue
            crop2 = gray[y2:y2+h2, x2:x2+w2]
            try:
                t2 = cv2.resize(crop2, (max(8,w1), max(8,h1)), interpolation=cv2.INTER_AREA)
                t1 = cv2.resize(crop1, (t2.shape[1], t2.shape[0]), interpolation=cv2.INTER_AREA)
            except Exception:
                continue
            e1 = cv2.Canny(t1,50,150); e2 = cv2.Canny(t2,50,150)
            res = cv2.matchTemplate(e1, e2, cv2.TM_CCOEFF_NORMED)
            _,maxVal,_,_ = cv2.minMaxLoc(res)
            std1 = np.std(t1)/255.0; std2 = np.std(t2)/255.0
            combined = 0.7*maxVal + 0.3*(1 - abs(std1-std2))
            if combined > ghost_conf:
                ghost_conf = float(combined); ghost_rect=(int(x2),int(y2),int(w2),int(h2))
    has_ghost = ghost_conf > 0.28
    return {"has_ghost": bool(has_ghost), "ghost_confidence": float(ghost_conf), "main_rect": main_rect, "ghost_rect": ghost_rect}

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(20,20))
    return int(len(faces)), [tuple(map(int, f)) for f in faces]

# ---------------- Text detection ----------------
def detect_text_boxes_craft(img):
    """
    Input: BGR np.uint8 image
    Output: list of polygons (list of 4 [x,y] points) and confidence placeholder
    """
    if not USE_CRAFT:
        return []
    # craft expects RGB PIL or path, but craft_text_detector accepts cv2 image
    try:
        prediction_result = craft.detect_text(img)  # returns dict structure
        # craft output has 'boxes' as numpy array Nx4x2 (clockwise), 'polys' similar, plus 'score_text'
        boxes = []
        if "boxes" in prediction_result:
            for b in prediction_result["boxes"]:
                box = [[int(round(float(pt[0]))), int(round(float(pt[1])))] for pt in b]
                boxes.append(normalize_box(box))
        else:
            # fallback to 'polys'
            if "polys" in prediction_result:
                for p in prediction_result["polys"]:
                    boxes.append(normalize_box(p))
        return boxes
    except Exception:
        return []

def detect_text_boxes_mser(img):
    # fallback detector using MSER
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Correct MSER initialization
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(5000)

    regions, _ = mser.detectRegions(gray)
    boxes = []

    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        if w * h < 500:
            continue
        boxes.append(normalize_box([[x, y], [x + w, y + h]]))

    # merge overlapping boxes (simple)
    merged = []
    for b in boxes:
        merged_flag = False
        for i, mb in enumerate(merged):
            if iou_box(b, mb) > 0.3:
                xs = [pt[0] for pt in mb + b]
                ys = [pt[1] for pt in mb + b]
                merged[i] = [
                    [min(xs), min(ys)],
                    [max(xs), min(ys)],
                    [max(xs), max(ys)],
                    [min(xs), max(ys)],
                ]
                merged_flag = True
                break
        if not merged_flag:
            merged.append(b)

    return merged


# ---------------- Recognition using TrOCR (or fallback) ----------------
def recognize_crop_trocr(bgr_crop):
    if not USE_TROCR:
        return ""
    try:
        img_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        pixel_values = trocr_processor(images=pil, return_tensors="pt").pixel_values.to(device)
        generated_ids = trocr_model.generate(pixel_values, max_length=128)
        preds = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return preds[0].strip() if preds else ""
    except Exception:
        return ""

def recognize_crop_paddle(bgr_crop):
    try:
        # paddle accepts BGR numpy
        res = paddle_ocr.ocr(bgr_crop)
        texts = []
        for block in res:
            for token in block:
                if len(token) >= 2:
                    info = token[1]
                    if isinstance(info, (list, tuple)) and len(info) >= 1:
                        texts.append(str(info[0]))
        return " ".join(texts).strip()
    except Exception:
        return ""

def recognize_crop(bgr_crop):
    # try TrOCR first (if available), then Paddle fallback
    txt = ""
    if USE_TROCR:
        txt = recognize_crop_trocr(bgr_crop)
    if (not txt or len(txt.strip()) == 0) and USE_PADDLE_FALLBACK:
        txt = recognize_crop_paddle(bgr_crop)
    return txt.strip()

# ---------------- Merge tokens & lines ----------------
def merge_boxes_and_recognize(img, boxes):
    tokens = []
    for b in boxes:
        # crop with safe bounds
        minx = max(min(pt[0] for pt in b), 0); miny = max(min(pt[1] for pt in b), 0)
        maxx = min(max(pt[0] for pt in b), img.shape[1]-1); maxy = min(max(pt[1] for pt in b), img.shape[0]-1)
        if maxx <= minx or maxy <= miny:
            continue
        crop = img[miny:maxy+1, minx:maxx+1]
        # try enhancements for recognition (grayscale + adaptive threshold) if needed
        rec = recognize_crop(crop)
        tokens.append({"text": rec, "confidence": 1.0 if rec else 0.0, "box": normalize_box(b)})
    # simple sort top->bottom
    tokens.sort(key=lambda x: min(pt[1] for pt in x["box"]) if x["box"] else 0)
    return tokens

def group_tokens_into_lines(tokens, y_tol=18):
    entries = []
    for t in tokens:
        ys = [pt[1] for pt in t["box"]]; xs = [pt[0] for pt in t["box"]]
        cy = int(sum(ys)/len(ys)) if ys else 0; cx = int(sum(xs)/len(xs)) if xs else 0
        entries.append({"token": t, "cy": cy, "cx": cx})
    entries.sort(key=lambda e: e["cy"])
    lines = []
    for e in entries:
        placed = False
        for ln in lines:
            if abs(e["cy"] - ln["cy_mean"]) <= y_tol:
                ln["tokens"].append(e["token"]); ln["ys"].append(e["cy"]); ln["xs"].append(e["cx"])
                ln["cy_mean"] = int(sum(ln["ys"]) / len(ln["ys"])); placed=True; break
        if not placed:
            lines.append({"tokens":[e["token"]], "ys":[e["cy"]], "xs":[e["cx"]], "cy_mean": e["cy"]})
    final=[]
    for ln in lines:
        texts = [t["text"] for t in ln["tokens"] if t.get("text")]
        joined = " ".join(texts).strip()
        confs = [t.get("confidence",0.0) for t in ln["tokens"]]
        avg_conf = float(sum(confs)/len(confs)) if confs else 0.0
        xs = [pt[0] for t in ln["tokens"] for pt in t["box"]]
        ys = [pt[1] for t in ln["tokens"] for pt in t["box"]]
        if xs and ys:
            minx,miny,maxx,maxy = min(xs),min(ys),max(xs),max(ys)
            union = [[int(minx),int(miny)],[int(maxx),int(miny)],[int(maxx),int(maxy)],[int(minx),int(maxy)]]
        else:
            union = [[0,0],[0,0],[0,0],[0,0]]
        final.append({"text": joined, "confidence": avg_conf, "box": union, "tokens": ln["tokens"]})
    return final

# ---------------- Field extraction ----------------
def fuzzy_label_match(text, variants, cutoff=0.7):
    s = (text or "").lower()
    for v in variants:
        if v in s: return True
    tokens = re.split(r'[\s:,\-]+', s)
    for t in tokens:
        if get_close_matches(t, variants, cutoff=cutoff): return True
    return False

def extract_fields_from_lines(lines):
    found = {}
    boxes = {}
    used = set()
    n = len(lines)
    # label-based first pass
    for i,ln in enumerate(lines):
        txt = ln["text"].strip()
        for field, variants in LABEL_VARIANTS.items():
            if fuzzy_label_match(txt, variants):
                if ":" in txt:
                    val = txt.split(":",1)[1].strip()
                else:
                    parts = txt.split()
                    pos = -1
                    for j,w in enumerate(parts):
                        if w.lower() in variants or get_close_matches(w.lower(), variants, cutoff=0.75):
                            pos = j; break
                    if pos >= 0 and pos+1 < len(parts):
                        val = " ".join(parts[pos+1:]).strip()
                    else:
                        val = lines[i+1]["text"].strip() if (i+1)<n else ""
                # basic postprocess
                if field == "DOB":
                    m = RE_DOB.search(val)
                    if m: val = m.group(0)
                    else: val = "".join(re.findall(r'\d', val)) or val
                if field in ("Phone","PinCode","IDNumber"):
                    digs = "".join(re.findall(r'\d', val))
                    if digs: val = digs
                if val:
                    found[field] = val
                    boxes[field] = ln["box"]
                    used.add(i)
                break
    # regex fallbacks for unlabeled
    for i,ln in enumerate(lines):
        if i in used: continue
        t = ln["text"]
        if "Email" not in found:
            m = RE_EMAIL.search(t)
            if m: found["Email"] = m.group(0); boxes["Email"]=ln["box"]; used.add(i); continue
        if "Phone" not in found:
            m = RE_PHONE.search(t)
            if m:
                digits = "".join(re.findall(r'\d', m.group(0)))
                if 7<=len(digits)<=15:
                    found["Phone"]=digits; boxes["Phone"]=ln["box"]; used.add(i); continue
        if "DOB" not in found:
            m = RE_DOB.search(t)
            if m: found["DOB"]=m.group(0); boxes["DOB"]=ln["box"]; used.add(i); continue
        if "IDNumber" not in found:
            m = RE_AADHAAR.search(t)
            if m: found["IDNumber"] = m.group(0).replace(" ",""); boxes["IDNumber"]=ln["box"]; used.add(i); continue
            m2 = RE_PAN.search(t)
            if m2: found["IDNumber"]=m2.group(0); boxes["IDNumber"]=ln["box"]; used.add(i); continue
        if "BloodGroup" not in found:
            m = RE_BLOOD.search(t)
            if m: found["BloodGroup"]=m.group(0).upper(); boxes["BloodGroup"]=ln["box"]; used.add(i); continue
        if "PinCode" not in found:
            m = RE_PIN.search(t)
            if m: found["PinCode"]=m.group(0); boxes["PinCode"]=ln["box"]; used.add(i); continue
    # name fallback: top-most alpha-heavy
    if "Name" not in found:
        for i,ln in enumerate(lines):
            if i in used: continue
            s = ln["text"].strip()
            if not s or "@" in s: continue
            alpha_ratio = sum(1 for ch in s if ch.isalpha()) / max(1,len(s))
            if alpha_ratio > 0.6 and len(s.split()) <= 7:
                found["Name"]=s; boxes["Name"]=ln["box"]; used.add(i); break
    # address fallback: long bottom lines
    if "Address" not in found:
        for i in range(len(lines)-1, -1, -1):
            if i in used: continue
            s = lines[i]["text"].strip()
            if len(s) > 25:
                found["Address"]=s; boxes["Address"]=lines[i]["box"]; used.add(i); break
    # ensure keys exist with empty string default
    final = {k: found.get(k,"") for k in TARGET_FIELDS}
    return final, boxes

# ---------------- Drawing & annotate ----------------
def draw_annotated_image(img, tokens, lines, field_boxes, face_rects, out_path):
    out = img.copy()
    # tokens boxes colored by confidence
    for tk in tokens:
        try:
            box = np.array(tk["box"], dtype=np.int32).reshape((-1,1,2))
            conf = float(tk.get("confidence", 0.0))
            color = (0,255,0) if conf >= 0.85 else ((0,255,255) if conf >= 0.6 else (0,0,255))
            cv2.polylines(out, [box], isClosed=True, color=color, thickness=2)
            x,y = box[0][0]
            text = tk["text"][:40]
            cv2.putText(out, text, (max(2,x), max(10,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        except Exception:
            pass
    # lines union boxes
    for ln in lines:
        try:
            a = tuple(ln["box"][0]); c = tuple(ln["box"][2])
            cv2.rectangle(out, a, c, (200,200,200), 1)
        except Exception:
            pass
    # field boxes labeled
    for fld,b in (field_boxes or {}).items():
        try:
            box = np.array(b, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(out, [box], isClosed=True, color=(255,0,0), thickness=3)
            x,y = box[0][0]
            cv2.putText(out, f"[{fld}]", (max(2,x), max(10,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
        except Exception:
            pass
    # faces
    for fr in face_rects or []:
        try:
            x,y,w,h = fr
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,128,0), 2)
        except Exception:
            pass
    try:
        cv2.imwrite(out_path, out)
    except Exception:
        pass
    return out_path

# ---------------- Main processing per image ----------------
def process_image(path, show_window=False, save_annotated=True, debug=False):
    if not os.path.exists(path):
        return {"status":"error", "message": f"file not found: {path}"}
    img = cv2.imread(path)
    if img is None:
        return {"status":"error", "message": f"failed to load: {path}"}
    # Quality checks
    blur_info = detect_blur_metrics(img)
    ghost_info = detect_ghost_via_template(img)
    face_count, face_rects = detect_faces(img)

    # Detect boxes (CRAFT preferred)
    boxes = []
    if USE_CRAFT:
        try:
            boxes = detect_text_boxes_craft(img)
        except Exception:
            boxes = []
    if not boxes:
        boxes = detect_text_boxes_mser(img)

    if debug:
        print(f"[DEBUG] Detected {len(boxes)} candidate boxes")

    # Merge overlapping & recognize each box (no resizing of main image)
    tokens = merge_boxes_and_recognize(img, boxes)
    lines = group_tokens_into_lines(tokens, y_tol=18)

    # Extract fields
    fields, field_boxes = extract_fields_from_lines(lines)

    # Final fallback: attempt to detect missing critical fields by scanning tokens
    if all(not v for v in fields.values()):
        for tk in tokens:
            t = tk["text"]
            if not fields["Email"]:
                m = RE_EMAIL.search(t)
                if m: fields["Email"] = m.group(0); field_boxes["Email"] = tk["box"]
            if not fields["Phone"]:
                m = RE_PHONE.search(t)
                if m:
                    digits = "".join(re.findall(r'\d', m.group(0)))
                    if 7 <= len(digits) <= 15:
                        fields["Phone"] = digits; field_boxes["Phone"] = tk["box"]
            if not fields["DOB"]:
                m = RE_DOB.search(t)
                if m: fields["DOB"] = m.group(0); field_boxes["DOB"] = tk["box"]
            if not fields["IDNumber"]:
                m = RE_PAN.search(t)
                if m: fields["IDNumber"] = m.group(0); field_boxes["IDNumber"] = tk["box"]
                else:
                    m2 = RE_AADHAAR.search(t)
                    if m2:
                        fields["IDNumber"] = m2.group(0).replace(" ", ""); field_boxes["IDNumber"] = tk["box"]
            if not fields["BloodGroup"]:
                m = RE_BLOOD.search(t)
                if m: fields["BloodGroup"] = m.group(0).upper(); field_boxes["BloodGroup"] = tk["box"]
            if not fields["PinCode"]:
                m = RE_PIN.search(t)
                if m: fields["PinCode"] = m.group(0); field_boxes["PinCode"] = tk["box"]
        # Name fallback
        if not fields["Name"]:
            for tk in tokens:
                s = tk["text"].strip()
                if not s or "@" in s: continue
                alpha_ratio = sum(1 for ch in s if ch.isalpha())/max(1,len(s))
                if alpha_ratio > 0.6 and len(s.split()) <= 7:
                    fields["Name"] = s; field_boxes["Name"] = tk["box"]; break

    # Ensure all target fields exist as keys (empty string if not found)
    final_fields = OrderedDict((k, fields.get(k, "")) for k in TARGET_FIELDS)

    # Prepare quality & ocr_data
    ocr_data = [{"text": tk["text"], "confidence": float(round(tk.get("confidence", 0.0),3)), "box": tk["box"]} for tk in tokens]
    quality = {
        "is_blurry": bool(blur_info["is_blurry"]),
        "blur_score": float(blur_info["lap_var"]),
        "fft_mean": float(blur_info["fft_mean"]),
        "ghost_image_detected": bool(ghost_info["has_ghost"]),
        "ghost_confidence": float(ghost_info["ghost_confidence"]),
        "face_count": int(face_count)
    }

    result = {
        "status": "success",
        "quality_check": quality,
        "ocr_data": ocr_data,
        "lines": lines,
        "fields": final_fields
    }

    # Annotate & save
    annotated_path = os.path.splitext(path)[0] + "_annotated.png"
    draw_annotated_image(img, tokens, lines, field_boxes, face_rects, annotated_path)

    # Terminal output: print clean fields dict (format A) and save JSON file per run (optional)
    print("\n=== FIELDS OUTPUT (clean dict) ===")
    print(safe_json(final_fields))

    # Also print a small summary
    print("\nGhost confidence: %.3f, face_count: %d, blur_score: %.2f" % (quality["ghost_confidence"], quality["face_count"], quality["blur_score"]))
    print("Annotated image saved:", annotated_path)

    return result

# ---------------- CLI ----------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        imgs = sys.argv[1:]
    else:
        imgs = ["Images/sampleee1.png","Images/sampleee2.png","Images/sampleee3.png"]
    all_results = {}
    for p in imgs:
        print("\n--- Processing:", p, "---")
        res = process_image(p, show_window=False, save_annotated=True, debug=True)
        all_results[os.path.basename(p)] = res
    # save aggregated JSON
    try:
        with open("hybrid_multi_results.json","w",encoding="utf-8") as f:
            f.write(safe_json(all_results))
        print("\nSaved combined JSON: hybrid_multi_results.json")
    except Exception as e:
        print("Failed to save combined JSON:", e)
