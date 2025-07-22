import React, { useState } from "react";
import axios from "axios";
import ResultCard from "./ResultCard";

export default function FileChecker({ addToHistory }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileCheck = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await axios.post("http://localhost:8000/predict-file", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      addToHistory({
        type: "File",
        name: file.name,
        predicted_label: response.data.predicted_label,
        malicious_probability: response.data.malicious_probability || 0,
      });
    } catch (err) {
      console.error(err);
      alert("Error analyzing file");
    }
    setLoading(false);
  };

  return (
    <div className="checker-card">
      <h2 className="checker-title">Check a File</h2>
      <input
        type="file"
        className="input-file"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleFileCheck} disabled={loading} className="btn-primary">
        {loading ? <span className="loader"></span> : "Analyze File"}
      </button>
      {result && <ResultCard data={result} />}
    </div>
  );
}