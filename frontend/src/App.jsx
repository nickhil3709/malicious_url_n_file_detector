import React, { useState } from "react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import Reports from "./pages/Reports";

export default function App() {
  const [activeView, setActiveView] = useState("dashboard"); // 'dashboard' | 'reports'
  const [activeTab, setActiveTab] = useState("url"); // passed into Dashboard to sync with sidebar
  const [history, setHistory] = useState([]); // [{type,name,predicted_label,malicious_probability,timestamp}]

  const addToHistory = (entry) => {
    const stamped = { ...entry, timestamp: new Date().toISOString() };
    setHistory((prev) => [stamped, ...prev].slice(0, 25)); // keep 25 globally; Dashboard shows top 5
  };

  const handleNav = (dest) => {
    if (dest === "url" || dest === "file") {
      setActiveTab(dest);
      setActiveView("dashboard");
    } else if (dest === "dashboard") {
      setActiveView("dashboard");
    } else if (dest === "reports") {
      setActiveView("reports");
    }
  };

  return (
    <div className="app-shell">
      <Sidebar activeView={activeView} activeTab={activeTab} onNavigate={handleNav} />
      <main className="main-content">
        {activeView === "dashboard" ? (
          <Dashboard
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            history={history}
            addToHistory={addToHistory}
          />
        ) : (
          <Reports history={history} />
        )}
      </main>
    </div>
  );
}