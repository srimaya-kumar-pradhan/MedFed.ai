import { useEffect, useState } from "react";
import { training, nodes } from "../api/client";
import type { NodeInfo } from "../api/client";
import { Icon } from "../components/Icon";

export function HistoryPage() {
  // The Clinical Portal's history is the user's own recent analyses.
  // In this prototype, the history is sourced from the most recent
  // training-run outputs, and the user can also re-analyze a sample image.
  const [runs, setRuns] = useState<Awaited<ReturnType<typeof training.list>>["runs"]>([]);
  const [nodeList, setNodeList] = useState<NodeInfo[]>([]);

  useEffect(() => {
    training.list().then((r) => setRuns(r.runs)).catch(() => undefined);
    nodes.list().then((r) => setNodeList(r.nodes)).catch(() => undefined);
  }, []);

  return (
    <div className="p-4 max-w-6xl space-y-3">
      <div>
        <h1 className="text-2xl">History</h1>
        <p className="text-sm text-ink-500 mt-0.5">
          Recent federated training rounds and active hospital nodes for this institution.
        </p>
      </div>

      <div className="card">
        <h2 className="text-lg mb-2">Recent training rounds</h2>
        {runs.length === 0 ? (
          <p className="text-sm text-ink-500">No training rounds recorded yet.</p>
        ) : (
          <table className="table">
            <thead>
              <tr><th>Run</th><th>Strategy</th><th>Rounds</th><th>Status</th><th>Started</th></tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td className="font-mono text-xs">{r.id.slice(0, 22)}…</td>
                  <td>{r.strategy}</td>
                  <td className="font-mono">{r.current_round} / {r.rounds}</td>
                  <td>
                    <span className={`badge ${r.status === "completed" ? "badge-good" : r.status === "running" ? "badge-warn" : r.status === "failed" ? "badge-bad" : "badge-neutral"}`}>{r.status}</span>
                  </td>
                  <td className="font-mono text-xs">{r.started_at?.slice(0, 19) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h2 className="text-lg mb-2">Active hospital nodes</h2>
        {nodeList.length === 0 ? (
          <p className="text-sm text-ink-500">No hospital nodes available.</p>
        ) : (
          <table className="table">
            <thead><tr><th>Node</th><th>Total samples</th><th>Train / Val / Test</th><th>Locality</th></tr></thead>
            <tbody>
              {nodeList.map((n) => (
                <tr key={n.node_id}>
                  <td className="font-mono">{n.node_id}</td>
                  <td className="font-mono">{n.total_samples.toLocaleString()}</td>
                  <td className="font-mono text-xs">{n.train_samples.toLocaleString()} / {n.val_samples.toLocaleString()} / {n.test_samples.toLocaleString()}</td>
                  <td><span className="badge-neutral"><Icon.Shield className="w-3 h-3" /> Isolated</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
