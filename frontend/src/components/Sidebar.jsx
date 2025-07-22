import React from "react";

export default function Sidebar({ activeView, activeTab, onNavigate }) {
  const isDashboard = activeView === "dashboard";
  return (
    <aside className="sidebar">
      <div className="sidebar-brand" onClick={() => onNavigate("dashboard")}>
        <span className="sidebar-logo">⚡</span> CyberGuard
      </div>
      <nav className="sidebar-nav">
        <button
          className={`sidebar-link ${isDashboard && activeTab === "url" ? "active" : ""}`}
          onClick={() => onNavigate("url")}
        >
          URL Checker
        </button>
        <button
          className={`sidebar-link ${isDashboard && activeTab === "file" ? "active" : ""}`}
          onClick={() => onNavigate("file")}
        >
          File Checker
        </button>
        <button
          className={`sidebar-link ${activeView === "reports" ? "active" : ""}`}
          onClick={() => onNavigate("reports")}
        >
          Reports
        </button>
      </nav>
    </aside>
  );
}