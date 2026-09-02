import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { nodes, models, training } from "../api/client";
import type { NodeInfo, ModelsList, TrainingJob } from "../api/client";

export function LocalDatasetPage() {
  const [nodeList, setNodeList] = useState<NodeInfo[]>([]);
  useEffect(() => { nodes.list().then((r) => setNodeList(r.nodes)).catch(() => undefined); }, []);

  return (
    <div className="p-4 max-w-5xl space-y-3">
      <div>
        <h1 className="text-2xl">Local Dataset</h1>
        <p className="text-sm text-ink-500 mt-0.5">Sample counts held in each hospital's local partition. Raw images never leave the node.</p>
      </div>
      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th>Node</th>
              <th>Total</th>
              <th>Train</th>
              <th>Val</th>
              <th>Test</th>
            </tr>
          </thead>
          <tbody>
            {nodeList.map((n) => (
              <tr key={n.node_id}>
                <td className="font-mono">{n.node_id}</td>
                <td className="font-mono">{n.total_samples.toLocaleString()}</td>
                <td className="font-mono">{n.train_samples.toLocaleString()}</td>
                <td className="font-mono">{n.val_samples.toLocaleString()}</td>
                <td className="font-mono">{n.test_samples.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function TrainingRunsPage() {
  const [runs, setRuns] = useState<TrainingJob[]>([]);
  useEffect(() => {
    training.list().then((r) => setRuns(r.runs)).catch(() => undefined);
  }, []);

  return (
    <div className="p-4 max-w-6xl space-y-3">
      <div>
        <h1 className="text-2xl">Training Runs</h1>
        <p className="text-sm text-ink-500 mt-0.5">All training runs ever executed by this orchestrator.</p>
      </div>
      <div className="card">
        {runs.length === 0 ? (
          <p className="text-sm text-ink-500">No training runs yet.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Run</th><th>Strategy</th><th>Privacy</th><th>Rounds</th><th>Status</th><th>Started</th><th>Ended</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td className="font-mono text-xs">{r.id}</td>
                  <td>{r.strategy}</td>
                  <td>{r.privacy}</td>
                  <td className="font-mono">{r.current_round} / {r.rounds}</td>
                  <td>
                    <span className={`badge ${r.status === "completed" ? "badge-good" : r.status === "running" ? "badge-warn" : r.status === "failed" ? "badge-bad" : "badge-neutral"}`}>{r.status}</span>
                  </td>
                  <td className="font-mono text-xs">{r.started_at?.slice(0, 19) ?? "—"}</td>
                  <td className="font-mono text-xs">{r.ended_at?.slice(0, 19) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export function PerformancePage() {
  const [list, setList] = useState<ModelsList | null>(null);
  useEffect(() => { models.list().then(setList).catch(() => undefined); }, []);

  return (
    <div className="p-4 max-w-5xl space-y-3">
      <div>
        <h1 className="text-2xl">Performance</h1>
        <p className="text-sm text-ink-500 mt-0.5">Validation metrics from real model versions. No fabricated numbers.</p>
      </div>
      <div className="card">
        {!list ? <p className="text-sm text-ink-500">Loading…</p> : (
          <table className="table">
            <thead>
              <tr><th>Version</th><th>Round</th><th>F1</th><th>ROC-AUC</th><th>Loss</th><th>Strategy</th><th>Privacy</th></tr>
            </thead>
            <tbody>
              {list.versions.map((v) => (
                <tr key={v.version}>
                  <td className="font-mono">{v.version}</td>
                  <td className="font-mono">{v.round ?? "—"}</td>
                  <td className="font-mono">{v.metrics.f1 === null ? "N/A" : v.metrics.f1.toFixed(4)}</td>
                  <td className="font-mono">{v.metrics.roc_auc === null ? "N/A" : v.metrics.roc_auc.toFixed(4)}</td>
                  <td className="font-mono">{v.metrics.loss === null ? "N/A" : v.metrics.loss.toFixed(4)}</td>
                  <td className="font-mono">{v.metadata.strategy ?? "—"}</td>
                  <td className="font-mono">{v.metadata.privacy ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export function NodeStatusPage() {
  const [nodeList, setNodeList] = useState<NodeInfo[]>([]);
  useEffect(() => { nodes.list().then((r) => setNodeList(r.nodes)).catch(() => undefined); }, []);

  return (
    <div className="p-4 max-w-5xl space-y-3">
      <div>
        <h1 className="text-2xl">Node Status</h1>
        <p className="text-sm text-ink-500 mt-0.5">Federation participants. Data locality is enforced at the storage layer.</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        {nodeList.map((n) => (
          <div key={n.node_id} className="card">
            <div className="flex items-center justify-between mb-2">
              <div className="font-mono">{n.node_id}</div>
              <span className="badge-good"><span className="dot dot-good" /> Connected</span>
            </div>
            <div className="text-sm text-ink-500">Total samples</div>
            <div className="font-mono text-lg">{n.total_samples.toLocaleString()}</div>
            <div className="mt-2 text-xs text-ink-500">
              Local training only · no cross-node data transfer.
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SecurityPage() {
  return (
    <div className="p-4 max-w-3xl space-y-3">
      <div>
        <h1 className="text-2xl">Security & Privacy</h1>
        <p className="text-sm text-ink-500 mt-0.5">How MedFed AI protects patient data in the federated loop.</p>
      </div>

      <div className="card">
        <h2 className="text-lg mb-2">Data flow</h2>
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] items-center gap-2 text-sm">
          <div className="border border-ink-200 px-2 py-2 rounded-sm">
            <div className="font-mono">Hospital</div>
            <div className="text-xs text-ink-500">Local training only</div>
          </div>
          <div className="text-center text-ink-500 text-xs">protected updates only</div>
          <div className="border border-ink-200 px-2 py-2 rounded-sm">
            <div className="font-mono">Orchestrator</div>
            <div className="text-xs text-ink-500">Never sees patient data</div>
          </div>
        </div>
        <p className="text-xs text-ink-500 mt-2">
          Patient data → Hospital infrastructure → never leaves. Only model parameter updates are transmitted.
        </p>
      </div>

      <div className="card">
        <h2 className="text-lg mb-2">Access control</h2>
        <table className="table">
          <thead><tr><th>Role</th><th>Permissions</th></tr></thead>
          <tbody>
            <tr><td>Doctor</td><td>Run inference, view history, view explanations</td></tr>
            <tr><td>Researcher</td><td>Start training, view training runs, evaluate models</td></tr>
            <tr><td>Institution Admin</td><td>Deploy models, manage nodes, configure institution</td></tr>
            <tr><td>Platform Admin</td><td>Full platform oversight</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
