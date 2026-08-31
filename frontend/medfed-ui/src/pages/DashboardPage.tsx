import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { system, nodes, models, training } from "../api/client";
import type { NodeInfo, TrainingJob } from "../api/client";
import { Icon } from "../components/Icon";

type ModelStatus = Awaited<ReturnType<typeof system.modelStatus>>;
type ModelsList   = Awaited<ReturnType<typeof models.list>>;

export function DashboardPage() {
  const { state, hasPermission } = useAuth();
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [nodeList, setNodeList]       = useState<NodeInfo[]>([]);
  const [modelList, setModelList]     = useState<ModelsList | null>(null);
  const [recentRuns, setRecentRuns]   = useState<TrainingJob[]>([]);

  useEffect(() => {
    system.modelStatus().then(setModelStatus).catch(() => undefined);
    if (hasPermission("view_nodes")) {
      nodes.list().then((r) => setNodeList(r.nodes)).catch(() => undefined);
    }
    if (hasPermission("view_model_registry")) {
      models.list().then(setModelList).catch(() => undefined);
    }
    if (hasPermission("view_training_runs")) {
      training.list().then((r) => setRecentRuns(r.runs.slice(0, 5))).catch(() => undefined);
    }
  }, [hasPermission]);

  const userName = state.status === "authed" ? state.user.full_name : "";
  const greeting = `Welcome, ${userName}`;

  return (
    <div className="p-4 space-y-4 max-w-6xl">
      <div>
        <h1 className="text-2xl">Dashboard</h1>
        <p className="text-sm text-ink-500 mt-0.5">{greeting}.</p>
      </div>

      {/* Status row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
        <Stat label="Model status" value={modelStatus?.available ? "Ready" : "Unavailable"} tone={modelStatus?.available ? "good" : "bad"} />
        <Stat label="Current model" value={modelStatus?.registry?.current_version ?? "—"} mono />
        <Stat label="Round" value={modelStatus?.registry?.round ? String(modelStatus.registry.round) : "N/A"} mono />
        <Stat label="Device" value={modelStatus?.device ?? "—"} mono />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-2">
        {/* Model panel */}
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg">Current model</h2>
            {modelStatus?.available ? (
              <span className="badge-good"><span className="dot dot-good" /> Ready</span>
            ) : (
              <span className="badge-bad"><span className="dot dot-bad" /> Unavailable</span>
            )}
          </div>
          {modelStatus?.registry ? (
            <div className="grid grid-cols-2 gap-2 text-sm">
              <Field label="Version" value={modelStatus.registry.current_version ?? "—"} mono />
              <Field label="Path" value={modelStatus.registry.path ?? "—"} mono small />
              <Field label="Architecture" value={String(modelStatus.registry.metadata?.architecture ?? "—")} mono />
              <Field label="Classes" value={String(modelStatus.registry.metadata?.num_classes ?? "—")} mono />
              <Field label="F1" value={fmt(modelStatus.registry.metrics?.f1)} mono />
              <Field label="ROC-AUC" value={fmt(modelStatus.registry.metrics?.roc_auc)} mono />
              <Field label="Strategy" value={String(modelStatus.registry.metadata?.strategy ?? "—")} mono />
              <Field label="Privacy" value={String(modelStatus.registry.metadata?.privacy ?? "—")} mono />
            </div>
          ) : (
            <p className="text-sm text-ink-500">Loading…</p>
          )}
        </div>

        {/* Quick actions */}
        <div className="card">
          <h2 className="text-lg mb-2">Quick actions</h2>
          <div className="flex flex-col gap-2">
            <Link to="/analyze" className="btn-accent"><Icon.Image /> Analyze image</Link>
            <Link to="/history" className="btn-secondary"><Icon.Clock /> View history</Link>
            {hasPermission("start_training") && (
              <Link to="/training" className="btn-secondary"><Icon.Network /> Start training</Link>
            )}
            {hasPermission("view_model_registry") && (
              <Link to="/models" className="btn-secondary"><Icon.Layers /> Model registry</Link>
            )}
          </div>
        </div>
      </div>

      {/* Federation / training context (only meaningful for research users) */}
      {hasPermission("view_nodes") && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
          <div className="card">
            <h2 className="text-lg mb-2">Hospital nodes</h2>
            {nodeList.length === 0 ? (
              <p className="text-sm text-ink-500">No nodes registered.</p>
            ) : (
              <table className="table">
                <thead>
                  <tr><th>Node</th><th>Samples</th><th>Locality</th></tr>
                </thead>
                <tbody>
                  {nodeList.map((n) => (
                    <tr key={n.node_id}>
                      <td className="font-mono">{n.node_id}</td>
                      <td className="font-mono">{n.total_samples.toLocaleString()}</td>
                      <td>
                        <span className="badge-neutral"><span className="dot dot-good" /> Isolated</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="card">
            <h2 className="text-lg mb-2">Recent training runs</h2>
            {recentRuns.length === 0 ? (
              <p className="text-sm text-ink-500">No training runs yet. Start one from the Federated Training page.</p>
            ) : (
              <table className="table">
                <thead>
                  <tr><th>Run</th><th>Strategy</th><th>Status</th><th>Rounds</th></tr>
                </thead>
                <tbody>
                  {recentRuns.map((r) => (
                    <tr key={r.id}>
                      <td className="font-mono text-xs">{r.id.slice(0, 18)}…</td>
                      <td>{r.strategy}</td>
                      <td>
                        <StatusBadge status={r.status} />
                      </td>
                      <td className="font-mono">{r.current_round} / {r.rounds}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* Model versions table (research only) */}
      {hasPermission("view_model_registry") && modelList && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg">Model versions</h2>
            <Link to="/models" className="text-sm text-accent hover:underline">View all</Link>
          </div>
          <table className="table">
            <thead>
              <tr><th>Version</th><th>Round</th><th>F1</th><th>ROC-AUC</th><th>Created</th><th>Status</th></tr>
            </thead>
            <tbody>
              {modelList.versions.map((v) => (
                <tr key={v.version}>
                  <td className="font-mono">{v.version}</td>
                  <td className="font-mono">{v.round ?? "—"}</td>
                  <td className="font-mono">{fmt(v.metrics.f1)}</td>
                  <td className="font-mono">{fmt(v.metrics.roc_auc)}</td>
                  <td className="font-mono text-xs">{v.created_at?.slice(0, 10) ?? "—"}</td>
                  <td>{v.version === modelList.current_version ? <span className="badge-good">Current</span> : <span className="badge-neutral">Archived</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function fmt(v: number | null | undefined) {
  return v === null || v === undefined ? "N/A" : v.toFixed(4);
}

function Stat({ label, value, tone, mono }: { label: string; value: string; tone?: "good" | "bad"; mono?: boolean }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className={`mt-1 text-xl ${mono ? "font-mono" : "font-medium"} ${tone === "good" ? "text-status-good" : tone === "bad" ? "text-status-bad" : "text-ink-900"}`}>
        {value}
      </div>
    </div>
  );
}

function Field({ label, value, mono, small }: { label: string; value: string; mono?: boolean; small?: boolean }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className={`${mono ? "font-mono" : ""} ${small ? "text-xs" : "text-sm"} text-ink-900 break-all`}>{value}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    pending:   "badge-neutral",
    running:   "badge-warn",
    completed: "badge-good",
    failed:    "badge-bad",
    stopped:   "badge-neutral",
  };
  return <span className={map[status] ?? "badge-neutral"}>{status}</span>;
}
