import type { ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { Icon } from "./Icon";
import { system } from "../api/client";
import { useEffect, useState } from "react";

type NavItem = { to: string; label: string; icon: keyof typeof Icon };

const CLINICAL_NAV: NavItem[] = [
  { to: "/dashboard",  label: "Dashboard",        icon: "Server" },
  { to: "/analyze",    label: "Analyze Image",    icon: "Image" },
  { to: "/history",    label: "History",          icon: "Clock" },
  { to: "/model",      label: "Model Information", icon: "Layers" },
  { to: "/help",       label: "Help",             icon: "Help" },
];

const RESEARCH_NAV: NavItem[] = [
  { to: "/dashboard",  label: "Dashboard",          icon: "Server" },
  { to: "/dataset",    label: "Local Dataset",      icon: "Layers" },
  { to: "/training",   label: "Federated Training", icon: "Network" },
  { to: "/runs",       label: "Training Runs",      icon: "Clock" },
  { to: "/models",     label: "Model Versions",     icon: "Layers" },
  { to: "/performance",label: "Performance",        icon: "Arrow" },
  { to: "/nodes",      label: "Node Status",        icon: "Network" },
  { to: "/security",   label: "Security",           icon: "Shield" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const { state, signOut, hasPermission } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const [modelLabel, setModelLabel] = useState<string>("—");

  useEffect(() => {
    system.modelStatus()
      .then((s) => {
        if (s.available && s.registry.current_version) {
          setModelLabel(`${s.registry.current_version}${s.registry.round ? ` · Round ${s.registry.round}` : ""}`);
        } else {
          setModelLabel("unavailable");
        }
      })
      .catch(() => setModelLabel("—"));
  }, []);

  if (state.status !== "authed") {
    return <>{children}</>;
  }

  const isResearch = hasPermission("start_training") || hasPermission("view_training_runs");
  const navItems = isResearch ? RESEARCH_NAV : CLINICAL_NAV;

  return (
    <div className="min-h-screen bg-paper text-ink-900 grid grid-cols-[240px_1fr]">
      {/* Sidebar */}
      <aside className="border-r border-ink-200 bg-paper flex flex-col">
        <div className="px-3 py-3 border-b border-ink-200">
          <Link to="/dashboard" className="block">
            <div className="text-base font-medium tracking-tight">MedFed AI</div>
            <div className="text-xs text-ink-500 mt-0.5">{isResearch ? "Research Portal" : "Clinical Portal"}</div>
          </Link>
        </div>

        <nav className="flex-1 py-2">
          {navItems.map((item) => {
            const I = Icon[item.icon];
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-3 py-2 text-sm border-l-2 ${
                    isActive
                      ? "border-ink-900 text-ink-900 bg-ink-50"
                      : "border-transparent text-ink-500 hover:text-ink-900 hover:bg-ink-50"
                  }`
                }
              >
                <I />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        <div className="border-t border-ink-200 p-3 text-xs">
          <div className="text-ink-500">Signed in as</div>
          <div className="mt-1 font-medium text-ink-900">{state.user.full_name}</div>
          <div className="text-ink-500">{state.user.role_label} · {state.user.hospital_id}</div>
          <button
            onClick={() => { signOut(); nav("/login"); }}
            className="mt-2 inline-flex items-center gap-1.5 text-ink-500 hover:text-ink-900"
          >
            <Icon.Logout />
            <span>Sign out</span>
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex flex-col min-h-screen">
        <header className="flex items-center justify-between border-b border-ink-200 px-4 py-2 bg-paper">
          <div className="text-sm text-ink-500">
            <span className="font-mono">{loc.pathname}</span>
          </div>
          <div className="text-xs text-ink-500 flex items-center gap-2">
            <span>Model:</span>
            <span className="font-mono text-ink-900">{modelLabel}</span>
          </div>
        </header>
        <div className="flex-1 anim-fade-in">{children}</div>
        <footer className="border-t border-ink-200 px-4 py-2 text-xs text-ink-400">
          MedFed AI · prototype · NIH Chest X-ray (14-class). Clinical decision remains with the qualified healthcare professional.
        </footer>
      </main>
    </div>
  );
}
