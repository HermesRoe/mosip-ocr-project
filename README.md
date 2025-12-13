# MOSIP Intelligent OCR & Identity Verification System

## 🚀 Project Overview
This project automates the extraction and verification of data from Government ID cards. It uses a hybrid AI pipeline combining **Computer Vision (OpenCV)** for structure detection and **Transformers (Microsoft TrOCR)** for high-accuracy text recognition. It is designed to streamline identity verification for MOSIP-based systems.

---

## 🏗️ Architectural Design
The solution follows a microservices architecture, separating the User Interface from the intensive Computer Vision processing.

### **System Components**
1.  **Client Layer (Frontend):**
    * Built with **React.js**.
    * Handles image upload, visualizes the OCR "Heatmap" (green bounding boxes), and manages the Verification Form.
2.  **Service Layer (Backend):**
    * Built with **FastAPI (Python)**.
    * Orchestrates data flow, validates files, and manages API endpoints.
3.  **Processing Layer (Computer Vision Module):**
    * **Preprocessing:** Uses **OpenCV** for "Smart Masking" (removing faces/QR codes) to prevent OCR noise.
    * **Text Detection:** Utilizes **CRAFT** (Character Region Awareness) or MSER for bounding box detection.
    * **Text Recognition:** Uses **Microsoft TrOCR** (Transformer OCR) for reading text.
    * **Parsing Logic:** Custom "Spatial Parsing" algorithm that groups words by line and associates values with labels (e.g., finding "Name" and looking below/right for the value).

---

## 🔄 Data Flow Structure
1.  **Input:** User uploads an image via the Frontend.
2.  **Sanitization:** The backend detects Faces and QR codes using Haar Cascades and masks them to isolate text.
3.  **Extraction:** The AI engine detects text regions, crops them, and feeds them to TrOCR.
4.  **Parsing:** The logic engine uses spatial geometry to map text to fields (Name, DOB, ID).
5.  **Response:** JSON data containing extracted text, confidence scores, and heatmap coordinates is sent back to the client.

---

## 🔌 API Documentation

### **1. Extract Data**
* **Endpoint:** `POST /extract`
* **Description:** Uploads an ID card image for processing.
* **Response:**
    ```json
    {
      "status": "success",
      "fields": {
        "Name": "Sarah Jenkins",
        "DOB": "12-05-1998",
        "IDNumber": "A987 6543 2109"
      },
      "ocr_data": [ ...list of detected boxes... ]
    }
    ```

### **2. Verify Data**
* **Endpoint:** `POST /verify`
* **Description:** Compares original AI extraction vs. user-submitted data.
* **Response:**
    ```json
    {
      "status": "success",
      "match_score": 100,
      "message": "Verified Perfect Match"
    }
    ```

---

## 🛠️ Installation & Setup

### **Backend (Python)**
1.  Navigate to `backend/src`.
2.  Create virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    ```
3.  Install dependencies:
    ```bash
    pip install fastapi uvicorn python-multipart opencv-python numpy pillow transformers torch torchvision craft-text-detector
    ```
4.  Run server:
    ```bash
    uvicorn main:app --reload
    ```

### **Frontend (React)**
1.  Navigate to `frontend/ocr-client`.
2.  Install packages:
    ```bash
    npm install
    ```
3.  Start client:
    ```bash
    npm start
    ```

---
*Note: This solution was developed for the MOSIP Hackathon.*
