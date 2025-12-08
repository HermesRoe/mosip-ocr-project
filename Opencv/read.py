import os
import sys
import re
import json
import math
import cv2
import numpy as np
from paddleocr import PaddleOCR
from difflib import get_close_matches

# ----------------------------
# Configuration
# ----------------------------
OCR_LANG = "en"
# initialize once
PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang=OCR_LANG)

TARGET_FIELDS = ["Name", "DOB", "Gender", "BloodGroup", "Phone", "Email", "Address", "IDNumber"]

LABEL_VARIANTS = {
    "Name": ["name", "full name", "first name", "given name"],
    "DOB": ["dob", "date of birth", "birth", "d.o.b"],
    "Gender": ["gender", "sex"],
    "BloodGroup": ["blood", "blood group", "bloodgrp", "bloodtype"],
    "Phone": ["phone", "phone number", "mobile", "mobile number", "contact"],
    "Email": ["email", "email id", "e-mail", "e mail"],
    "Address": ["address", "address line", "addr", "residence"],
    "IDNumber": ["id", "id no", "id number", "identification", "card no", "aadhar", "pan"]
}

RE_EMAIL = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', flags=re.I)
RE_PHONE = re.compile(r'(\+?\d[\d\-\s]{6,}\d)')
RE_DOB = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
RE_AADHAAR = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
RE_PAN = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b', flags=re.I)
RE_BLOOD = re.compile(r'\b(?:A|B|AB|O)[+-]\b', flags=re.I)

# ----------------------------
# Helpers
# ----------------------------
def safe_json(obj):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
    return json.dumps(obj, default=conv, indent=2)

def iou_box(a, b):
    ax1 = min(pt[0] for pt in a); ay1 = min(pt[1] for pt in a)
    ax2 = max(pt[0] for pt in a); ay2 = max(pt[1] for pt in a)
    bx1 = min(pt[0] for pt in b); by1 = min(pt[1] for pt in b)
    bx2 = max(pt[0] for pt in b); by2 = max(pt[1] for pt in b)
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def normalize_box(box):
    if not box:
        return [[0,0],[0,0],[0,0],[0,0]]
    pts = []
    for pt in box:
        try:
            x = int(round(float(pt[0]))); y = int(round(float(pt[1])))
        except Exception:
            x, y = 0, 0
        pts.append([x, y])
    if len(pts) == 2:
        (x0,y0),(x1,y1) = pts
        return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    if len(pts) < 4:
        while len(pts) < 4:
            pts.append(pts[-1] if pts else [0,0])
    return pts[:4]

# ----------------------------
# Ghost detection (face-based)
# ----------------------------
def detect_ghost_improved(img_color):
    """Return (has_ghost:bool, face_count:int, ghost_conf:float, main_rect, ghost_rect)."""
    try:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False, 0, 0.0, None, None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=4, minSize=(20,20))
    faces = sorted(list(faces), key=lambda f: f[2]*f[3], reverse=True) if len(faces) > 0 else []
    face_count = len(faces)
    if face_count >= 2:
        (x1,y1,w1,h1),(x2,y2,w2,h2) = faces[0], faces[1]
        main_rect = (int(x1),int(y1),int(w1),int(h1))
        ghost_rect = (int(x2),int(y2),int(w2),int(h2))
        try:
            ghost_crop = img_gray[y2:y2+h2, x2:x2+w2]
            ghost_contrast = float(np.std(ghost_crop))/255.0
        except Exception:
            ghost_contrast = 0.0
        size_ratio = (w2*h2)/max(1.0, (w1*h1))
        dist = math.dist((x1+w1/2, y1+h1/2),(x2+w2/2, y2+h2/2))
        norm_dist = dist / max(1.0, w1)
        ghost_conf = float(np.clip((size_ratio*(1-ghost_contrast))/0.5, 0.0, 1.0)) if (0.08 <= size_ratio <= 0.6 and norm_dist <= 1.8) else 0.0
        has_ghost = ghost_conf > 0.2
        return bool(has_ghost), int(face_count), float(ghost_conf), main_rect, ghost_rect
    if face_count == 1:
        x,y,w,h = faces[0]
        return False, int(face_count), 0.0, (int(x),int(y),int(w),int(h)), None
    return False, 0, 0.0, None, None

# ----------------------------
# Preprocess for OCR
# ----------------------------
def preprocess_for_ocr(img, enhance_handwriting=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bf = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(bf)
    den = cv2.fastNlMeansDenoising(cl, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if enhance_handwriting:
        thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        return thr
    return den

# ----------------------------
# Safe parse for PaddleOCR results (and scaling)
# ----------------------------
def safe_parse_paddle_result(result, scale=1.0):
    """
    Accept variety of possible PaddleOCR outputs and normalize to list of tokens:
    [{'text':..., 'confidence':..., 'box': [[x,y],...]}]
    """
    parsed = []

    # If result is None or empty:
    if not result:
        return parsed

    # Typical result variant: list of blocks, each block a list of tokens [ [box, (text, conf)], ... ]
    # But Paddle may sometimes return different nested shapes. We'll attempt multiple safe interpretations.
    try:
        # direct safe path if it matches expected nested lists:
        for block in result:
            if isinstance(block, (list, tuple)):
                # If block looks like a token (box, (text, conf)), wrap it as a single token-block
                # But usually block is a list of tokens
                # We'll iterate tokens inside block
                for token in block:
                    # token commonly is [box, (text, conf)]
                    if not isinstance(token, (list, tuple)):
                        continue
                    box_raw = token[0] if len(token) >= 1 else None
                    info = token[1] if len(token) >= 2 else ""
                    box = normalize_box(box_raw) if box_raw is not None else [[0,0],[0,0],[0,0],[0,0]]
                    if scale != 1.0:
                        inv = 1.0 / scale
                        box = [[int(round(pt[0] * inv)), int(round(pt[1] * inv))] for pt in box]
                    text = ""
                    conf = 0.0
                    try:
                        if isinstance(info, (list, tuple)):
                            if len(info) >= 1 and info[0] is not None:
                                text = str(info[0])
                            if len(info) >= 2:
                                try: conf = float(info[1])
                                except Exception: conf = 0.0
                        else:
                            text = str(info)
                    except Exception:
                        text = str(info) if info is not None else ""
                        conf = 0.0
                    parsed.append({"text": text.strip(), "confidence": float(conf), "box": box})
            elif isinstance(block, dict):
                # Some Paddle variants might return dicts per item
                # try to read common keys
                txt = block.get("text") or block.get("words") or block.get("sentence") or ""
                conf = block.get("confidence") or block.get("score") or 0.0
                box = block.get("box") or block.get("bbox") or block.get("points") or None
                box = normalize_box(box) if box is not None else [[0,0],[0,0],[0,0],[0,0]]
                if scale != 1.0:
                    inv = 1.0/scale
                    box = [[int(round(pt[0]*inv)), int(round(pt[1]*inv))] for pt in box]
                parsed.append({"text": str(txt).strip(), "confidence": float(conf or 0.0), "box": box})
            else:
                # unknown block type — ignore
                continue
    except Exception:
        # Last-resort: try a simpler parse
        try:
            for item in result:
                # item might be (box, text) or (text, conf)
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[0], (list, tuple)) and isinstance(item[1], (list, tuple)):
                        # assume (box, (text, conf))
                        box = normalize_box(item[0])
                        info = item[1]
                        text = info[0] if len(info) >= 1 else ""
                        conf = float(info[1]) if len(info) >= 2 else 0.0
                        parsed.append({"text": str(text).strip(), "confidence": conf, "box": box})
                    elif isinstance(item[0], str):
                        parsed.append({"text": str(item[0]).strip(), "confidence": float(item[1]) if isinstance(item[1], (int,float)) else 0.0, "box": [[0,0],[0,0],[0,0],[0,0]]})
        except Exception:
            pass

    return parsed

# ----------------------------
# Merge tokens from multiple passes
# ----------------------------
def merge_tokens(token_lists, iou_threshold=0.65):
    all_tokens = [t for lst in token_lists for t in lst]
    merged = []
    used = [False] * len(all_tokens)
    for i, t in enumerate(all_tokens):
        if used[i]: continue
        box_i = t["box"]
        text_i = t["text"]
        conf_i = t["confidence"]
        group_texts = [text_i]
        group_conf = [conf_i]
        group_boxes = [box_i]
        used[i] = True
        for j in range(i+1, len(all_tokens)):
            if used[j]: continue
            t2 = all_tokens[j]
            try:
                if iou_box(box_i, t2["box"]) >= iou_threshold:
                    used[j] = True
                    group_texts.append(t2["text"])
                    group_conf.append(t2["confidence"])
                    group_boxes.append(t2["box"])
            except Exception:
                continue
        best_idx = max(range(len(group_texts)), key=lambda k: (len(group_texts[k])>0, group_conf[k]))
        merged_text = group_texts[best_idx]
        merged_conf = max(group_conf) if group_conf else 0.0
        xs = [pt[0] for b in group_boxes for pt in b]
        ys = [pt[1] for b in group_boxes for pt in b]
        if xs and ys:
            minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
            union_box = [[int(minx),int(miny)],[int(maxx),int(miny)],[int(maxx),int(maxy)],[int(minx),int(maxy)]]
        else:
            union_box = [[0,0],[0,0],[0,0],[0,0]]
        merged.append({"text": merged_text, "confidence": float(merged_conf), "box": union_box})
    merged.sort(key=lambda x: min(pt[1] for pt in x["box"]) if x["box"] else 0)
    return merged

# ----------------------------
# Group merged tokens into lines
# ----------------------------
def group_tokens_into_lines(merged_tokens, y_tol=18):
    entries = []
    for t in merged_tokens:
        box = t["box"]
        if not box:
            continue
        ys = [pt[1] for pt in box]; xs = [pt[0] for pt in box]
        cy = int(sum(ys)/len(ys)); cx = int(sum(xs)/len(xs))
        entries.append({"token": t, "cy": cy, "cx": cx})
    entries.sort(key=lambda e: e["cy"])
    lines = []
    for e in entries:
        placed = False
        for ln in lines:
            if abs(e["cy"] - ln["cy_mean"]) <= y_tol:
                ln["tokens"].append(e["token"])
                ln["ys"].append(e["cy"]); ln["xs"].append(e["cx"])
                ln["cy_mean"] = int(sum(ln["ys"]) / len(ln["ys"]))
                placed = True; break
        if not placed:
            lines.append({"tokens":[e["token"]], "ys":[e["cy"]], "xs":[e["cx"]], "cy_mean": e["cy"]})
    final_lines = []
    for ln in lines:
        tokens = ln["tokens"]
        texts = [tk["text"] for tk in tokens if tk.get("text")]
        joined = " ".join(texts).strip()
        confs = [tk.get("confidence",0.0) for tk in tokens]
        avg_conf = float(sum(confs)/len(confs)) if confs else 0.0
        xs = [pt[0] for tk in tokens for pt in tk["box"]]
        ys = [pt[1] for tk in tokens for pt in tk["box"]]
        if xs and ys:
            minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
            union_box = [[int(minx),int(miny)],[int(maxx),int(miny)],[int(maxx),int(maxy)],[int(minx),int(maxy)]]
        else:
            union_box = [[0,0],[0,0],[0,0],[0,0]]
        final_lines.append({"text": joined, "confidence": avg_conf, "box": union_box, "tokens": tokens})
    return final_lines

# ----------------------------
# Field extraction (rule-based)
# ----------------------------
def fuzzy_label_match(text, variants, cutoff=0.7):
    s = text.lower()
    for v in variants:
        if v in s:
            return True
    tokens = re.split(r'[\s:,\-]+', s)
    for t in tokens:
        if get_close_matches(t, variants, cutoff=cutoff):
            return True
    return False

def extract_fields_from_lines(lines):
    found = {}
    boxes = {}
    used = set()
    n = len(lines)
    for i, ln in enumerate(lines):
        txt = ln["text"].strip()
        for field, variants in LABEL_VARIANTS.items():
            if fuzzy_label_match(txt, variants):
                if ":" in txt:
                    val = txt.split(":",1)[1].strip()
                else:
                    parts = txt.split()
                    label_pos = -1
                    for j,w in enumerate(parts):
                        if w.lower() in variants or get_close_matches(w.lower(), variants, cutoff=0.75):
                            label_pos = j; break
                    if label_pos >= 0 and label_pos+1 < len(parts):
                        val = " ".join(parts[label_pos+1:]).strip()
                    else:
                        val = lines[i+1]["text"].strip() if (i+1) < n else ""
                if field == "DOB":
                    m = RE_DOB.search(val)
                    if m: val = m.group(0)
                    else: val = "".join(re.findall(r'\d', val)) or val
                if field == "Phone":
                    digits = "".join(re.findall(r'\d', val))
                    if digits: val = digits
                if field == "IDNumber":
                    val = "".join(re.findall(r'\w', val)) or val
                if val:
                    found[field] = val
                    boxes[field] = ln["box"]
                    used.add(i)
                break
    # left-right splits
    for i, ln in enumerate(lines):
        if i in used: continue
        txt = ln["text"]
        parts = [p.strip() for p in re.split(r'\s{2,}', txt) if p.strip()]
        if len(parts) >= 2:
            left, right = parts[0], " ".join(parts[1:])
            for field, variants in LABEL_VARIANTS.items():
                if fuzzy_label_match(left.lower(), variants) and field not in found:
                    found[field] = right; boxes[field] = ln["box"]; used.add(i); break
    # regex fallbacks
    for i, ln in enumerate(lines):
        if i in used: continue
        t = ln["text"]
        if "Email" not in found:
            m = RE_EMAIL.search(t)
            if m: found["Email"] = m.group(0); boxes["Email"] = ln["box"]; used.add(i); continue
        if "Phone" not in found:
            m = RE_PHONE.search(t)
            if m:
                digits = "".join(re.findall(r'\d', m.group(0)))
                if 7 <= len(digits) <= 15:
                    found["Phone"] = digits; boxes["Phone"] = ln["box"]; used.add(i); continue
        if "DOB" not in found:
            m = RE_DOB.search(t)
            if m: found["DOB"] = m.group(0); boxes["DOB"] = ln["box"]; used.add(i); continue
        if "IDNumber" not in found:
            m = RE_AADHAAR.search(t)
            if m: found["IDNumber"] = m.group(0).replace(" ",""); boxes["IDNumber"] = ln["box"]; used.add(i); continue
            m2 = RE_PAN.search(t)
            if m2: found["IDNumber"] = m2.group(0); boxes["IDNumber"] = ln["box"]; used.add(i); continue
        if "BloodGroup" not in found:
            m = RE_BLOOD.search(t)
            if m: found["BloodGroup"] = m.group(0).upper(); boxes["BloodGroup"] = ln["box"]; used.add(i); continue
    # name & address fallback
    if "Name" not in found:
        for i, ln in enumerate(lines):
            if i in used: continue
            s = ln["text"].strip()
            if not s or "@" in s: continue
            alpha_ratio = sum(1 for ch in s if ch.isalpha()) / max(1, len(s))
            if alpha_ratio > 0.6 and len(s.split()) <= 6:
                found["Name"] = s; boxes["Name"] = ln["box"]; used.add(i); break
    if "Address" not in found:
        for i in range(len(lines)-1, -1, -1):
            if i in used: continue
            s = lines[i]["text"].strip()
            if len(s) > 25:
                found["Address"] = s; boxes["Address"] = lines[i]["box"]; used.add(i); break
    out = {k: found.get(k, "") for k in TARGET_FIELDS}
    return out, boxes

# ----------------------------
# Draw on original image
# ----------------------------
def draw_on_original(img_orig, merged_tokens, lines, field_boxes):
    out = img_orig.copy()
    for tk in merged_tokens:
        box = tk["box"]
        conf = tk["confidence"]
        color = (0,255,0) if conf >= 0.85 else ((0,255,255) if conf >= 0.6 else (0,0,255))
        try:
            pts = np.array(box, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            tx, ty = pts[0][0]
            cv2.putText(out, tk["text"][:40], (max(tx,2), max(ty-6,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        except Exception:
            pass
    for ln in lines:
        try:
            pts = np.array(ln["box"], dtype=np.int32).reshape((-1,1,2))
            cv2.rectangle(out, tuple(pts[0][0]), tuple(pts[2][0]), (200,200,200), 1)
        except Exception:
            pass
    for fld, b in (field_boxes or {}).items():
        if not b: continue
        try:
            pts = np.array(b, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(out, [pts], isClosed=True, color=(255,0,0), thickness=3)
            tx, ty = pts[0][0]
            cv2.putText(out, f"[{fld}]", (max(tx,2), max(ty-6,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
        except Exception:
            pass
    return out

# ----------------------------
# Helper to call OCR safely (predict() preferred)
# ----------------------------
def call_paddle_ocr(image):
    """
    Call PaddleOCR predict() if available, else ocr().
    Return raw result object (whatever Paddle returns) or None on failure.
    """
    try:
        # prefer predict
        if hasattr(PADDLE_OCR, "predict"):
            return PADDLE_OCR.predict(image)
        else:
            return PADDLE_OCR.ocr(image)
    except Exception as e:
        # try fallback to ocr if predict failed
        try:
            return PADDLE_OCR.ocr(image)
        except Exception:
            return None

# ----------------------------
# Main multi-pass pipeline (robust)
# ----------------------------
def process_image(image_path, show_window=True, save_annotated=True, debug=False):
    if not os.path.exists(image_path):
        return {"status":"error","message":f"file not found: {image_path}"}
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        return {"status":"error","message":f"failed to load: {image_path}"}

    # Quality checks
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blur = lap_var < 100.0
    edge_score = float(np.sum(cv2.Canny(gray,100,200)==255)/(gray.shape[0]*gray.shape[1]))
    has_ghost, face_count, ghost_conf, main_rect, ghost_rect = detect_ghost_improved(img_orig)

    # Prepare images for multi-pass
    prepped = preprocess_for_ocr(img_orig, enhance_handwriting=True)
    upscale = cv2.resize(img_orig, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    token_lists = []

    # PASS 1: original
    raw1 = call_paddle_ocr(img_orig)
    t1 = safe_parse_paddle_result(raw1, scale=1.0) if raw1 is not None else []
    token_lists.append(t1)
    if debug: print("PASS1 tokens:", len(t1))

    # PASS 2: upscaled (map back coordinates with scale=2)
    raw2 = call_paddle_ocr(upscale)
    t2 = safe_parse_paddle_result(raw2, scale=2.0) if raw2 is not None else []
    token_lists.append(t2)
    if debug: print("PASS2 tokens:", len(t2))

    # PASS 3: preprocessed (thresholded/CLAHE)
    raw3 = call_paddle_ocr(prepped)
    t3 = safe_parse_paddle_result(raw3, scale=1.0) if raw3 is not None else []
    token_lists.append(t3)
    if debug: print("PASS3 tokens:", len(t3))

    # If absolutely no tokens across passes
    total_tokens = sum(len(x) for x in token_lists)
    if total_tokens == 0:
        try:
            cv2.imwrite(os.path.splitext(image_path)[0] + "_prepped_debug.png", prepped)
            cv2.imwrite(os.path.splitext(image_path)[0] + "_upscaled_debug.png", upscale)
            if debug:
                print("Wrote debug images for inspection:",
                      os.path.splitext(image_path)[0] + "_prepped_debug.png",
                      os.path.splitext(image_path)[0] + "_upscaled_debug.png")
        except Exception:
            pass
        return {"status":"error","message":"No OCR tokens detected in any pass. See debug images."}

    # Merge, group, extract
    merged = merge_tokens(token_lists, iou_threshold=0.6)
    lines = group_tokens_into_lines(merged, y_tol=18)
    fields, boxes = extract_fields_from_lines(lines)

    # Fallback heuristics if all empty
    if all(not v for v in fields.values()):
        for tk in merged:
            t = tk["text"]
            if not fields["Email"]:
                m = RE_EMAIL.search(t)
                if m: fields["Email"] = m.group(0); boxes["Email"]=tk["box"]
            if not fields["Phone"]:
                m = RE_PHONE.search(t)
                if m:
                    digits = "".join(re.findall(r'\d', m.group(0)))
                    if 7 <= len(digits) <= 15:
                        fields["Phone"] = digits; boxes["Phone"]=tk["box"]
            if not fields["DOB"]:
                m = RE_DOB.search(t)
                if m: fields["DOB"] = m.group(0); boxes["DOB"] = tk["box"]
            if not fields["IDNumber"]:
                m = RE_PAN.search(t)
                if m: fields["IDNumber"] = m.group(0); boxes["IDNumber"]=tk["box"]
                else:
                    m2 = RE_AADHAAR.search(t)
                    if m2: fields["IDNumber"] = m2.group(0).replace(" ", ""); boxes["IDNumber"] = tk["box"]
            if not fields["BloodGroup"]:
                m = RE_BLOOD.search(t)
                if m: fields["BloodGroup"] = m.group(0).upper(); boxes["BloodGroup"]=tk["box"]
        if not fields["Name"]:
            for tk in merged:
                s = tk["text"].strip()
                if not s or "@" in s: continue
                alpha_ratio = sum(1 for ch in s if ch.isalpha())/max(1,len(s))
                if alpha_ratio > 0.6 and len(s.split()) <= 6:
                    fields["Name"] = s; boxes["Name"] = tk["box"]; break

    final_fields = {k: fields.get(k, "") for k in TARGET_FIELDS}
    ocr_data = [{"text": tk["text"], "confidence": float(round(tk["confidence"],3)), "box": tk["box"]} for tk in merged]
    quality = {
        "is_blurry": bool(is_blur),
        "blur_score": float(lap_var),
        "ghost_image_detected": bool(has_ghost),
        "ghost_confidence": float(ghost_conf),
        "face_count": int(face_count),
        "edge_score": float(edge_score)
    }

    result_json = {"status":"success", "quality_check": quality, "ocr_data": ocr_data, "lines": lines, "fields": final_fields}

    annotated = draw_on_original(img_orig, merged, lines, boxes)

    if save_annotated:
        outp = os.path.splitext(image_path)[0] + "_annotated.png"
        try:
            cv2.imwrite(outp, annotated)
        except Exception:
            pass

    if show_window:
        try:
            cv2.imshow("Annotated - " + os.path.basename(image_path), annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass

    if all(not v for v in final_fields.values()):
        print("\nDEBUG: Fields empty. OCR tokens (text, conf):")
        for tk in ocr_data:
            print(f"- '{tk['text']}' (conf={tk['confidence']}) box={tk['box']}")
        print("\nPaste token texts here and I'll refine rules.")

    print("\n--- Extracted fields ---")
    for k,v in final_fields.items():
        print(f"{k}: {v}")

    return result_json

# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        imgs = sys.argv[1:]
    else:
        default_dir = "images" if os.path.isdir("images") else "."
        imgs = [os.path.join(default_dir, nm) for nm in ("sampleee1.png","sampleee2.png","sampleee3.png")]

    all_results = {}
    for p in imgs:
        if not os.path.exists(p):
            print("Skipping missing:", p)
            continue
        print("\nProcessing:", p)
        res = process_image(p, show_window=True, save_annotated=True, debug=True)
        all_results[os.path.basename(p)] = res

    try:
        with open("ocr_multi_pass_results.json", "w", encoding="utf-8") as f:
            f.write(safe_json(all_results))
        print("\nSaved results to ocr_multi_pass_results.json")
    except Exception as e:
        print("Failed to save JSON:", e)
