import { useState, useEffect } from "react";
import type { FormEvent } from "react";
import { training, models, nodes } from "../api/client";
import type { NodeInfo, TrainingJob, ModelsList } from "../api/client";
import { Icon } from "../components/Icon";

export function TrainingPage() {
  const [form, setForm] = useState({
    strategy: "fedavg",
    privacy: "none",
    rounds: 3,
    local_epochs: 1,
    batch_size: 16,
    lr: 0.0001,
    mu: 0.01,
    max_batches: 15,
    seed: 42,
    hospital_nodes: ["Hospital_A", "Hospital_B", "Hospital_C"] as string[],
  });
  const [confirming, setConfirming] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [currentJob, setCurrentJob] = useState<TrainingJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nodeList, setNodeList] = useState<NodeInfo[]>([]);
  const [modelList, setModelList] = useState<ModelsList | null>(null);

  useEffect(() => {
    nodes.list().then((r) => setNodeList(r.nodes)).catch(() => undefined);
    models.list().then(setModelList).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!currentJob || (currentJob.status !== "running" && currentJob.status !== "pending")) return;
    const t = setInterval(() => {
      training.get(currentJob.id).then(setCurrentJob).catch(() => undefined);
    }, 2500);
    return () => clearInterval(t);
  }, [currentJob?.id, currentJob?.status]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!confirming) {
      setConfirming(true);
      return;
    }
    setSubmitting(true);
    try {
      const job = await training.start({ ...form, confirm: true });
      setCurrentJob(job);
      setConfirming(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
    } finally {
      setSubmitting(false);
    }
  };

  const stop = async () => {
    if (!currentJob) return;
    try {
      await training.stop(currentJob.id);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="p-4 max-w-6xl space-y-3">
      <div>
        <h1 className="text-2xl">Federated Training</h1>
        <p className="text-sm text-ink-500 mt-0.5">
          Configure and start a federated training round. Training is explicit, asynchronous, and never runs on startup.
        </p>
      </div>

      {/* Federation diagram */}
      <div className="card">
        <h2 className="text-lg mb-3">Federation topology</h2>
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] items-center gap-3">
          <div className="space-y-2">
            {nodeList.length === 0 ? (
              <p className="text-sm text-ink-500">Loading nodes…</p>
            ) : nodeList.map((n) => (
              <div key={n.node_id} className="border border-ink-200 px-2 py-2 rounded-sm flex items-center justify-between">
                <div>
                  <div className="font-mono text-sm">{n.node_id}</div>
                  <div className="text-xs text-ink-500">Local training · {n.total_samples.toLocaleString()} samples</div>
                </div>
                <span className="badge-neutral"><span className="dot dot-good" /> Isolated</span>
              </div>
            ))}
          </div>
          <div className="flex flex-col items-center justify-center text-xs text-ink-500">
            <Icon.Arrow className="rotate-90 md:rotate-0" />
            <div className="my-1 text-center max-w-[200px]">Protected model parameter updates only. Patient data never leaves the hospital.</div>
            <Icon.Arrow className="rotate-90 md:rotate-180" />
          </div>
          <div className="border border-ink-200 px-2 py-2 rounded-sm">
            <div className="font-mono text-sm">Federated Orchestrator</div>
            <div className="text-xs text-ink-500 mt-1">Aggregates model updates · never sees patient data</div>
            {modelList && (
              <div className="mt-2 text-xs">
                <div>Current: <span className="font-mono">{modelList.current_version}</span></div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="card">
          <h2 className="text-lg mb-2">Configure run</h2>
          <form onSubmit={submit} className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <SelectField label="Strategy" value={form.strategy} onChange={(v) => setForm({ ...form, strategy: v })} options={["fedavg", "fedprox", "fed-fibavg"]} />
              <SelectField label="Privacy" value={form.privacy} onChange={(v) => setForm({ ...form, privacy: v })} options={["none", "opacus", "opacus+prime"]} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <NumField label="Rounds" value={form.rounds} onChange={(v) => setForm({ ...form, rounds: v })} min={1} max={50} />
              <NumField label="Local epochs" value={form.local_epochs} onChange={(v) => setForm({ ...form, local_epochs: v })} min={1} max={20} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <NumField label="Batch size" value={form.batch_size} onChange={(v) => setForm({ ...form, batch_size: v })} min={1} max={128} />
              <NumField label="Max batches" value={form.max_batches} onChange={(v) => setForm({ ...form, max_batches: v })} min={1} max={500} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <NumField label="Learning rate" value={form.lr} onChange={(v) => setForm({ ...form, lr: v })} step={1e-5} />
              <NumField label="FedProx mu" value={form.mu} onChange={(v) => setForm({ ...form, mu: v })} step={0.01} />
            </div>

            {error && (
              <div className="border border-status-bad text-status-bad text-sm px-2 py-1.5 rounded-sm">{error}</div>
            )}

            {confirming ? (
              <div className="border border-status-warn bg-paper p-3 rounded-sm space-y-2">
                <div className="text-sm">
                  You are about to start a federated training round. This may use GPU/CPU resources and take several minutes.
                </div>
                <div className="flex gap-2">
                  <button type="button" className="btn-secondary" onClick={() => setConfirming(false)}>Cancel</button>
                  <button type="submit" className="btn-accent" disabled={submitting}>
                    {submitting ? "Starting…" : "Start Training"}
                  </button>
                </div>
              </div>
            ) : (
              <button type="submit" className="btn-primary w-full"><Icon.Play /> Start Training (confirm)</button>
            )}
          </form>
        </div>

        {/* Live status */}
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg">Live status</h2>
            {currentJob?.status === "running" && (
              <button className="btn-secondary" onClick={stop}>Stop</button>
            )}
          </div>
          {!currentJob ? (
            <p className="text-sm text-ink-500">No training job has been started in this session.</p>
          ) : (
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-ink-500">Job</span>
                <span className="font-mono text-xs">{currentJob.id}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-500">Status</span>
                <StatusBadge status={currentJob.status} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-500">Strategy</span>
                <span className="font-mono">{currentJob.strategy}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-500">Round</span>
                <span className="font-mono">{currentJob.current_round} / {currentJob.rounds}</span>
              </div>
              <div>
                <div className="h-2 bg-ink-100 rounded-sm overflow-hidden">
                  <div className="h-full bg-accent transition-all" style={{ width: `${currentJob.progress_pct}%` }} />
                </div>
                <div className="text-xs text-ink-500 mt-1 text-right font-mono">{currentJob.progress_pct.toFixed(1)}%</div>
              </div>
              {currentJob.per_round.length > 0 && (
                <div className="mt-2">
                  <div className="label">Per-round metrics</div>
                  <table className="table">
                    <thead><tr><th>R</th><th>F1</th><th>AUC</th></tr></thead>
                    <tbody>
                      {currentJob.per_round.map((r, i) => (
                        <tr key={i}>
                          <td className="font-mono">{r.round}</td>
                          <td className="font-mono">{r.global_macro_f1 !== null ? r.global_macro_f1.toFixed(4) : "—"}</td>
                          <td className="font-mono">{r.global_roc_auc !== null ? r.global_roc_auc.toFixed(4) : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <div>
      <div className="label mb-1">{label}</div>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

function NumField({ label, value, onChange, min, max, step }: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number }) {
  return (
    <div>
      <div className="label mb-1">{label}</div>
      <input
        className="input"
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
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
