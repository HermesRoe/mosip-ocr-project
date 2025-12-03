import React, { useState, useRef, useEffect, useCallback } from 'react'; // <--- Added useCallback
import QRCode from "react-qr-code"; 
import { uploadImageToBackend, verifyDataWithBackend } from './api';

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  
  const [ocrData, setOcrData] = useState(null);
  const [userEdits, setUserEdits] = useState({}); 
  const [verificationResult, setVerificationResult] = useState(null);

  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setStatus("");
      setOcrData(null);
      setVerificationResult(null);
      setUserEdits({});
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setStatus("Please select a file first.");
      return;
    }
    setLoading(true);
    setStatus("Uploading...");

    try {
      const data = await uploadImageToBackend(selectedFile);
      setOcrData(data);
      
      const initialEdits = {};
      data.ocr_data.forEach(item => {
        initialEdits[item.field] = item.text;
      });
      setUserEdits(initialEdits);

      setStatus("Success! Heatmap generated.");
    } catch (error) {
      setStatus("Error: Could not upload.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, newValue) => {
    setUserEdits(prev => ({
      ...prev,
      [field]: newValue
    }));
  };

  const handleVerify = async () => {
    setLoading(true);
    setStatus("Verifying...");
    try {
      const result = await verifyDataWithBackend(ocrData, userEdits);
      setVerificationResult(result);
      setStatus("Verification Complete!");
    } catch (error) {
      setStatus("Error during verification.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // --- UPDATED DRAWING LOGIC TO FIX WARNING ---
  // We wrap this function in useCallback so React knows it doesn't change randomly
  const drawHeatmap = useCallback(() => {
    if (ocrData && imageRef.current && canvasRef.current) {
      const img = imageRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ocrData.ocr_data.forEach((item) => {
        const [x, y, w, h] = item.box;
        let color = item.confidence > 0.9 ? '#00ff00' : 'orange';
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.2;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1.0;
      });
    }
  }, [ocrData]); // Only re-create this function if ocrData changes

  // Now we can safely add drawHeatmap to the dependency array
  useEffect(() => { drawHeatmap(); }, [drawHeatmap]);

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md overflow-hidden p-6 mt-10 mb-10">
      <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">Upload ID Card</h2>
      
      <div className="mb-6 flex justify-center">
        <input 
          type="file" accept="image/*" onChange={handleFileChange} 
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
      </div>

      <button 
        onClick={handleUpload} disabled={loading}
        className={`w-full py-2 px-4 rounded-md text-white font-bold ${loading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}`}
      >
        {loading ? "Processing..." : "Upload & Scan"}
      </button>

      {status && (
        <p className={`mt-4 text-center text-sm font-medium ${status.includes("Error") ? "text-red-600" : "text-green-600"}`}>
          {status}
        </p>
      )}

      {ocrData && (
        <div className="mt-8 border-t pt-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Extracted Data & Heatmap</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="border p-2 rounded bg-gray-50 relative">
              <p className="font-semibold mb-2 text-center">Heatmap Visualization</p>
              <div className="relative inline-block w-full">
                <img ref={imageRef} src={previewUrl} alt="Original" className="w-full rounded shadow block" onLoad={drawHeatmap} />
                <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
              </div>
            </div>

            <div className="space-y-4">
              <p className="font-semibold mb-2">Editable Fields</p>
              {ocrData.ocr_data.map((item, index) => (
                <div key={index} className="flex flex-col">
                  <label className="text-sm text-gray-600 font-bold">{item.field}</label>
                  <div className="flex items-center gap-2">
                    <input 
                      type="text" 
                      value={userEdits[item.field] || ""}
                      onChange={(e) => handleInputChange(item.field, e.target.value)}
                      className="border p-2 rounded w-full focus:ring-2 focus:ring-blue-500 outline-none"
                    />
                    <span className={`text-xs font-bold px-2 py-1 rounded ${item.confidence > 0.9 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                      {(item.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}

              {!verificationResult ? (
                <button 
                  onClick={handleVerify}
                  disabled={loading}
                  className="w-full mt-6 bg-green-600 text-white py-3 rounded hover:bg-green-700 font-bold transition"
                >
                  {loading ? "Verifying..." : "Verify Data"}
                </button>
              ) : (
                <div className={`mt-6 p-4 border rounded text-center 
                  ${verificationResult.match_score === 100 ? 'bg-green-50 border-green-400' : 'bg-yellow-50 border-yellow-400'}`}>
                  
                  <h4 className={`font-bold text-lg mb-2 ${verificationResult.match_score === 100 ? 'text-green-800' : 'text-yellow-800'}`}>
                    {verificationResult.match_score === 100 ? "✅ Verified Perfect Match" : "⚠️ Data Changed"}
                  </h4>
                  
                  <p className="text-gray-700 font-medium">Match Score: {verificationResult.match_score}%</p>
                  
                  {verificationResult.match_score === 100 && (
                    <div className="mt-4 flex flex-col items-center">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide font-bold">Digital Credential</p>
                      <div className="bg-white p-2 rounded shadow-sm">
                        <QRCode 
                          value={`MOSIP-VERIFIED:${ocrData.ocr_data[2].text}`} 
                          size={100} 
                        />
                      </div>
                      <p className="text-[10px] text-gray-400 mt-2">Scan to verify on blockchain</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;