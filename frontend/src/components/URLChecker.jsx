import React, { useState } from "react";
import axios from "axios";
import ResultCard from "./ResultCard";

export default function URLChecker({ addToHistory }) {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCheck = async () => {
    if (!url) return;
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/predict-url", { url });
      setResult(response.data);
      addToHistory({
        type: "URL",
        name: url,
        predicted_label: response.data.predicted_label,
        malicious_probability: response.data.malicious_probability || 0,
      });
    } catch (err) {
      console.error(err);
      alert("Error checking URL");
    }
    setLoading(false);
  };

  return (
    <div className="checker-card">
      <h2 className="checker-title">Check a URL</h2>
      <input
        type="text"
        placeholder="Enter URL"
        className="input-field"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
      />
      <button onClick={handleCheck} disabled={loading} className="btn-primary">
        {loading ? <span className="loader"></span> : "Check URL"}
      </button>
      {result && <ResultCard data={result} />}
    </div>
  );
}