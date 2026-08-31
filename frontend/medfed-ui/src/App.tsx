import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AppShell } from "./components/AppShell";
import { LoginPage } from "./pages/LoginPage";
import { DashboardPage } from "./pages/DashboardPage";
import { AnalyzePage } from "./pages/AnalyzePage";
import { HistoryPage } from "./pages/HistoryPage";
import { ModelInfoPage } from "./pages/ModelInfoPage";
import { HelpPage } from "./pages/HelpPage";
import { TrainingPage } from "./pages/TrainingPage";
import { ModelsRegistryPage } from "./pages/ModelsRegistryPage";
import {
  LocalDatasetPage, TrainingRunsPage, PerformancePage,
  NodeStatusPage, SecurityPage,
} from "./pages/ResearchPages";

function Protected({ children }: { children: React.ReactElement }) {
  const { state } = useAuth();
  const loc = useLocation();
  if (state.status === "loading") {
    return <div className="p-4 text-sm text-ink-500">Loading…</div>;
  }
  if (state.status === "anon") {
    return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  }
  return children;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="*"
        element={
          <Protected>
            <AppShell>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/analyze" element={<AnalyzePage />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/model" element={<ModelInfoPage />} />
                <Route path="/help" element={<HelpPage />} />
                <Route path="/training" element={<TrainingPage />} />
                <Route path="/runs" element={<TrainingRunsPage />} />
                <Route path="/models" element={<ModelsRegistryPage />} />
                <Route path="/dataset" element={<LocalDatasetPage />} />
                <Route path="/performance" element={<PerformancePage />} />
                <Route path="/nodes" element={<NodeStatusPage />} />
                <Route path="/security" element={<SecurityPage />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </AppShell>
          </Protected>
        }
      />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}
