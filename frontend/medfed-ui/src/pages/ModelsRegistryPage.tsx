import { useEffect, useState } from "react";
import { models } from "../api/client";
import type { ModelsList } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Icon } from "../components/Icon";

export function ModelsRegistryPage() {
  const { hasPermission } = useAuth();
  const [list, setList] = useState<ModelsList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deploying, setDeploying] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  const reload = () => models.list().then(setList).catch((e) => setError(String(e?.detail ?? e)));

  useEffect(() => { reload(); }, []);

  const deploy = async (version: string) => {
    setError(null);
    setDeploying(version);
    try {
      await models.deploy(version, true);
      setConfirming(null);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Deploy failed");
    } finally {
      setDeploying(null);
    }
  };

  const canDeploy = hasPermission("deploy_model");

  return (
    <div className="p-4 max-w-6xl space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl">Model Registry</h1>
          <p className="text-sm text-ink-500 mt-0.5">Every version produced by the federation. Deployment is human-approved.</p>
        </div>
        <button className="btn-secondary" onClick={reload}><Icon.Refresh /> Refresh</button>
      </div>

      {error && (
        <div className="border border-status-bad text-status-bad text-sm px-3 py-2 rounded-sm">{error}</div>
      )}

      {list ? (
        <div className="card">
          <table className="table">
            <thead>
              <tr>
                <th>Version</th>
                <th>Round</th>
                <th>F1</th>
                <th>ROC-AUC</th>
                <th>Strategy</th>
                <th>Privacy</th>
                <th>Created</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {list.versions.length === 0 ? (
                <tr><td colSpan={9} className="text-ink-500 text-sm py-3">No model versions registered. Start a training run to create one.</td></tr>
              ) : list.versions.map((v) => (
                <tr key={v.version}>
                  <td className="font-mono">{v.version}</td>
                  <td className="font-mono">{v.round ?? "—"}</td>
                  <td className="font-mono">{fmt(v.metrics.f1)}</td>
                  <td className="font-mono">{fmt(v.metrics.roc_auc)}</td>
                  <td className="font-mono">{v.metadata.strategy ?? "—"}</td>
                  <td className="font-mono">{v.metadata.privacy ?? "—"}</td>
                  <td className="font-mono text-xs">{v.created_at?.slice(0, 10) ?? "—"}</td>
                  <td>
                    {v.version === list.current_version
                      ? <span className="badge-good">Current</span>
                      : <span className="badge-neutral">Archived</span>}
                  </td>
                  <td>
                    {canDeploy && v.version !== list.current_version ? (
                      confirming === v.version ? (
                        <div className="flex gap-1.5">
                          <button className="btn-accent" disabled={deploying === v.version} onClick={() => deploy(v.version)}>
                            {deploying === v.version ? "Deploying…" : "Confirm"}
                          </button>
                          <button className="btn-secondary" onClick={() => setConfirming(null)}>Cancel</button>
                        </div>
                      ) : (
                        <button className="btn-secondary" onClick={() => setConfirming(v.version)}>Deploy</button>
                      )
                    ) : (
                      <span className="text-xs text-ink-400">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-ink-500">Loading…</p>
      )}
    </div>
  );
}

function fmt(v: number | null) {
  return v === null || v === undefined ? "N/A" : v.toFixed(4);
}
