import React, { useEffect, useState } from "react";
import URLChecker from "../components/URLChecker";
import FileChecker from "../components/FileChecker";
import DetectionChart from "../components/DetectionChart";

export default function Dashboard({ activeTab, setActiveTab, history, addToHistory }) {
  const [localTab, setLocalTab] = useState(activeTab);

  // Sync when sidebar changes activeTab
  useEffect(() => {
    setLocalTab(activeTab);
  }, [activeTab]);

  const handleTab = (tab) => {
    setLocalTab(tab);
    setActiveTab(tab); // update parent (App)
  };

  // show top 5 only in dashboard history
  const top5 = history.slice(0, 5);

  return (
    <div className="dashboard">
      <h1 className="dashboard-title">CyberGuard</h1>
      <p className="dashboard-subtitle">Malicious URL & File Detection</p>

      <div className="tabs">
        <button
          className={`tab-btn ${localTab === "url" ? "active" : ""}`}
          onClick={() => handleTab("url")}
        >
          URL Checker
        </button>
        <button
          className={`tab-btn ${localTab === "file" ? "active" : ""}`}
          onClick={() => handleTab("file")}
        >
          File Checker
        </button>
      </div>

      <div className="checker-container">
        {localTab === "url" ? (
          <URLChecker addToHistory={addToHistory} />
        ) : (
          <FileChecker addToHistory={addToHistory} />
        )}
      </div>

      <div className="history-section">
        <h3 className="history-title">Recent Checks</h3>
        {top5.length === 0 ? (
          <p className="history-empty">No recent checks</p>
        ) : (
          <ul className="history-list">
            {top5.map((item, index) => (
              <li
                key={index}
                className={`history-item ${item.predicted_label === 1 ? "malicious" : "safe"}`}
              >
                <strong>{item.type}:</strong> {item.name} <br />
                <span className="prob">Probability: {item.malicious_probability}%</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <DetectionChart history={top5} />
    </div>
  );
}