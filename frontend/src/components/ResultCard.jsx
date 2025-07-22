import React from "react";

export default function ResultCard({ data }) {
  const isMal = data.predicted_label === 1;
  return (
    <div className={`result-card ${isMal ? "malicious" : "safe"}`}>
      <h3 className="result-title">Detection Result</h3>
      <p><strong>Prediction:</strong> {isMal ? "Malicious" : "Safe"}</p>
      {data.malicious_probability !== undefined && (
        <p><strong>Malicious Probability:</strong> {data.malicious_probability}%</p>
      )}
    </div>
  );
}