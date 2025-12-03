import axios from 'axios';

// The URL where Member B's server will run
const API_BASE_URL = "http://localhost:8000";

// 1. UPLOAD IMAGE API
export const uploadImageToBackend = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  // --- REAL CODE (Commented out until Member B is ready) ---
  // const response = await axios.post(`${API_BASE_URL}/extract`, formData, {
  //   headers: { "Content-Type": "multipart/form-data" }
  // });
  // return response.data;
  
  // --- MOCK CODE (Active) ---
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        quality_check: { is_blurry: false, blur_score: 120, ghost_image_detected: false },
        ocr_data: [
          { field: "Name", text: "John Doe", confidence: 0.98, box: [20, 100, 150, 30] },
          { field: "DOB", text: "15-08-1995", confidence: 0.85, box: [20, 150, 120, 30] },
          { field: "ID No", text: "A1234567", confidence: 0.99, box: [20, 200, 150, 30] }
        ]
      });
    }, 1500);
  });
};

// 2. VERIFY DATA API
export const verifyDataWithBackend = async (originalData, userEdits) => {
  // --- REAL CODE (Commented out) ---
  // const response = await axios.post(`${API_BASE_URL}/verify`, {
  //   original: originalData,
  //   corrected: userEdits
  // });
  // return response.data;

  // --- MOCK CODE (Active) ---
  return new Promise((resolve) => {
    setTimeout(() => {
      // LOGIC: Compare the Original OCR vs. What User Typed
      let totalFields = originalData.ocr_data.length;
      let matchingFields = 0;

      originalData.ocr_data.forEach(item => {
        const originalText = item.text;
        const userText = userEdits[item.field];
        
        // Check if they match
        if (originalText === userText) {
          matchingFields++;
        }
      });

      // Calculate the score
      const calculatedScore = Math.round((matchingFields / totalFields) * 100);

      resolve({
        status: calculatedScore === 100 ? "success" : "partial_match",
        match_score: calculatedScore, 
        message: calculatedScore === 100 ? "Data Verified Successfully!" : "Data Modified by User."
      });
    }, 1000);
  });
};