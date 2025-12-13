import React from 'react';
import FileUpload from './components/FileUpload';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 py-10">
      <header className="text-center mb-10">
        <h1 className="text-4xl font-extrabold text-blue-900 tracking-tight">
          MOSIP OCR System
        </h1>
        <p className="mt-2 text-lg text-gray-600">
          Upload an ID card to extract and verify data.
        </p>
      </header>
      
      <main>
        <FileUpload />
      </main>
    </div>
  );
}

export default App;