import React, { useState } from 'react';
import api from '../services/api';

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post('/api/analyze', formData);
    setResult(res.data);
  };

  const handleChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  return (
    <div>
      <input type="file" onChange={handleChange} />
      {preview && <img src={preview} alt="preview" className="w-64" />}
      <button onClick={handleUpload} className="bg-blue-500 text-white p-2">Analyze</button>
      {result && <div>Prediction: {JSON.stringify(result)}</div>}
    </div>
  );
};

export default ImageUpload;
