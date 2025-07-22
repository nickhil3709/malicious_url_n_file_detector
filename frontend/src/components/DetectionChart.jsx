import React, { useMemo } from "react";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

export default function DetectionChart({ history }) {
  const { safeCount, maliciousCount } = useMemo(() => {
    let safe = 0, mal = 0;
    history.forEach((h) => {
      if (h.predicted_label === 1) mal += 1; else safe += 1;
    });
    return { safeCount: safe, maliciousCount: mal };
  }, [history]);

  const data = {
    labels: ["Safe", "Malicious"],
    datasets: [
      {
        data: [safeCount, maliciousCount],
        backgroundColor: ["#3bb143", "#d62828"],
        hoverBackgroundColor: ["#2d8e34", "#a61e1e"],
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "bottom" },
    },
  };

  return (
    <div className="chart-container">
      <h3 className="chart-title">Detection Summary</h3>
      {safeCount === 0 && maliciousCount === 0 ? (
        <p className="chart-empty">No data yet</p>
      ) : (
        <Pie data={data} options={options} />
      )}
    </div>
  );
}