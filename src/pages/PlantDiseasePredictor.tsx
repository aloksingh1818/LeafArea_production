import React, { useState } from 'react';

const API_URL = 'http://localhost:8000/predict';

const PlantDiseasePredictor = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(e.target.files?.[0] || null);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || 'Prediction failed.');
      }
    } catch (err) {
      setError('Network error.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-4 bg-white rounded shadow mt-8">
      <h2 className="text-2xl font-bold mb-4 text-green-700">Plant Disease Predictor</h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button
          type="submit"
          className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          disabled={!selectedFile || loading}
        >
          {loading ? 'Predicting...' : 'Predict Disease'}
        </button>
      </form>
      {result && (
        <div className="mt-4 p-3 bg-green-50 rounded border border-green-200">
          <div className="font-semibold">Prediction: <span className="text-green-800">{result.predicted_class}</span></div>
          <div>Confidence: {(result.confidence * 100).toFixed(2)}%</div>
        </div>
      )}
      {error && (
        <div className="mt-4 p-3 bg-red-50 rounded border border-red-200 text-red-700">{error}</div>
      )}
    </div>
  );
};

export default PlantDiseasePredictor;
