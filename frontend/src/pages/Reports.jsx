import React, { useMemo } from "react";
import DetectionChart from "../components/DetectionChart";

export default function Reports({ history }) {
  const stats = useMemo(() => {
    let safe = 0, mal = 0, avg = 0;
    if (!history.length) return { safe, mal, avg: 0 };
    history.forEach((h) => {
      if (h.predicted_label === 1) mal += 1; else safe += 1;
      avg += Number(h.malicious_probability || 0);
    });
    avg = +(avg / history.length).toFixed(1);
    return { safe, mal, avg };
  }, [history]);

  return (
    <div className="reports">
      <h1 className="reports-title">Detection Reports</h1>
      <p className="reports-subtitle">Summary of recent URL & file analyses.</p>

      <div className="reports-stats">
        <div className="reports-stat safe">Safe: {stats.safe}</div>
        <div className="reports-stat mal">Malicious: {stats.mal}</div>
        <div className="reports-stat avg">Avg Probability: {stats.avg}%</div>
      </div>

      <DetectionChart history={history} />

      <table className="reports-table">
        <thead>
          <tr>
            <th>Type</th>
            <th>Name</th>
            <th>Prediction</th>
            <th>Malicious %</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {history.length === 0 ? (
            <tr><td colSpan="5" className="history-empty">No data</td></tr>
          ) : (
            history.map((item, i) => (
              <tr key={i} className={item.predicted_label === 1 ? "malicious-row" : "safe-row"}>
                <td>{item.type}</td>
                <td title={item.name}>{item.name.length > 32 ? item.name.slice(0, 32) + "…" : item.name}</td>
                <td>{item.predicted_label === 1 ? "Malicious" : "Safe"}</td>
                <td>{item.malicious_probability}%</td>
                <td>{new Date(item.timestamp).toLocaleString()}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}