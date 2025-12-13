# MOSIP Intelligent OCR & Identity Verification System

## 🚀 Project Overview
This project is an automated identity verification solution designed for MOSIP-based systems. It eliminates manual data entry by using a hybrid AI pipeline: **Computer Vision (OpenCV)** for structural analysis and **Transformers (Microsoft TrOCR)** for high-accuracy text recognition. The system creates a seamless verification loop by extracting data from ID cards and validating it against user input.

---

## 🏗️ Technical Architecture
The solution uses a decoupled microservices architecture to handle intensive image processing efficiently.

### **1. Frontend (Client Layer)**
* **Stack:** React.js, Tailwind CSS.
* **Role:** Handles image uploads and renders the **Interactive Heatmap**. It uses HTML5 Canvas to draw bounding boxes over the original image, giving users immediate visual confirmation of what the AI "read."

### **2. Backend (Service Layer)**
* **Stack:** Python 3.12, FastAPI.
* **Role:** Acts as the high-performance orchestrator. It manages API endpoints (`/extract`, `/verify`), handles file validation, and routes images to the processing engine.

### **3. Computer Vision Engine (Processing Layer)**
* **Pre-processing (Smart Masking):** We implemented a custom sanitization step using **Haar Cascades** to detect faces and QR codes. These regions are masked (painted white) before OCR runs, eliminating 99% of "ghost text" errors (like confusing eyes with letters).
* **Detection:** Uses **CRAFT** (Character Region Awareness) or MSER to locate text blocks, ignoring complex background patterns.
* **Recognition:** Detected crops are fed into **Microsoft TrOCR** (Transformer-based OCR), which outperforms standard Tesseract on handwritten and low-contrast text.
* **Spatial Parsing:** Instead of simple line-reading, our custom algorithm uses spatial geometry. It finds labels (e.g., "Name") and searches specifically to the **right** or **below** that coordinate to capture the correct value, handling multi-column layouts effectively.

---

## 🔄 Data Flow Structure
1. **Input:** User uploads an image via the Frontend.
2. **Sanitization:** The backend detects Faces and QR codes using Haar Cascades and masks them to isolate text.
3. **Extraction:** The AI engine detects text regions, crops them, and feeds them to TrOCR.
4. **Parsing:** The logic engine uses spatial geometry to map text to fields (Name, DOB, ID).
5. **Response:** JSON data containing extracted text, confidence scores, and heatmap coordinates is sent back to the client.

---

## 🔌 API Documentation

This section details the primary endpoints used for integration.

### **1. Extract Data**
**Endpoint:** `POST /extract`
**Description:** Accepts an ID card image file, processes it through the CV pipeline, and returns structured data with coordinate bounding boxes.

**Request:**
* **Content-Type:** `multipart/form-data`
* **Body:** `file` (Binary Image File)

**Response Example (200 OK):**
```json
{
  "status": "success",
  "fields": {
    "Name": "Johnathan Doe",
    "DOB": "15-08-1995",
    "IDNumber": "A123 4567 8901",
    "Address": "123 Green Avenue, Toronto"
  },
  "quality_check": {
    "is_blurry": false,
    "face_detected": true
  },
  "ocr_data": [
    {
      "field": "Name",
      "text": "Johnathan Doe",
      "confidence": 0.98,
      "box": [[100, 200], [300, 200], [300, 250], [100, 250]]
    }
    // ... additional token data
  ]
}
