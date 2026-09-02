import { useEffect, useState } from "react";
import { system, nodes } from "../api/client";
import type { NodeInfo } from "../api/client";
import { Icon } from "./Icon";

// ──────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────
interface ClassDistribution {
  class_counts: Record<string, number>;
  total_samples: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
}

interface DataDistributionResponse {
  classes: string[];
  nodes: Record<string, ClassDistribution>;
  global_class_totals: Record<string, number>;
  global_total_samples: number;
  ks_statistic: number;
  skew_verdict: string;
  ks_by_node: Record<string, number>;
}

const SKEW_LABELS: Record<string, string> = {
  normal: "Normal — class frequencies are roughly uniform",
  moderate: "Moderate skew — some classes are over-represented",
  high: "High skew — significant class imbalance across hospitals",
};

const SKEW_TONE: Record<string, "good" | "warn" | "bad"> = {
  normal: "good",
  moderate: "warn",
  high: "bad",
};

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────
function maxCount(dist: ClassDistribution, classes: string[]): number {
  let m = 0;
  for (const c of classes) {
    m = Math.max(m, dist.class_counts[c] ?? 0);
  }
  return m || 1;
}

// ──────────────────────────────────────────────────────────────────────
// Stacked horizontal bar for a single node
// ──────────────────────────────────────────────────────────────────────
function NodeBar({
  node_id,
  dist,
  classes,
}: {
  node_id: string;
  dist: ClassDistribution;
  classes: string[];
}) {
  const total = dist.total_samples;
  const maxC = maxCount(dist, classes);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="font-mono text-ink-900">{node_id}</span>
        <span className="text-xs text-ink-500 font-mono">
          {total.toLocaleString()} samples
        </span>
      </div>
      <div className="flex h-5 rounded-sm overflow-hidden border border-ink-200">
        {classes.map((c, i) => {
          const cnt = dist.class_counts[c] ?? 0;
          const pct = total > 0 ? (cnt / total) * 100 : 0;
          // Simple hue rotation per class for distinguishability on grayscale
          const hue = (i * 47 + 180) % 360;
          return (
            <div
              key={c}
              className="relative group"
              style={{
                width: `${pct}%`,
                minWidth: pct > 0 ? 2 : 0,
                backgroundColor: `hsl(${hue}, 55%, 38%)`,
              }}
            >
              <span className="absolute inset-0 flex items-center justify-center text-[9px] text-white font-mono pointer-events-none overflow-hidden whitespace-nowrap">
                {pct >= 7 && cnt > 0 ? `${cnt}` : ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Main component
// ──────────────────────────────────────────────────────────────────────
export function DataDistributionPage() {
  const [data, setData] = useState<DataDistributionResponse | null>(null);
  const [nodesList, setNodesList] = useState<NodeInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      system.modelStatus().catch(() => null),
      nodes.list()
        .then((r) => setNodesList(r.nodes))
        .catch(() => undefined),
    ])
      .then(() =>
        fetch("/api/data/distribution")
          .then((res) => {
            if (!res.ok) throw new Error(res.statusText);
            return res.json();
          })
          .then(setData)
          .catch((e) => setError(e.message))
          .finally(() => setLoading(false))
      )
      .catch(() => setError("Failed to load data distribution"));
  }, []);

  if (loading) {
    return (
      <div className="p-4 max-w-6xl">
        <h1 className="text-2xl">Data Distribution</h1>
        <p className="text-sm text-ink-500 mt-0.5">Loading data distribution…</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-4 max-w-6xl space-y-3">
        <h1 className="text-2xl">Data Distribution</h1>
        <div className="border border-status-bad text-status-bad text-sm px-3 py-2 rounded-sm">
          {error ?? "No data available"}
        </div>
      </div>
    );
  }

  const { classes, nodes: nodeData, ks_statistic, skew_verdict, ks_by_node } = data;

  return (
    <div className="p-4 max-w-6xl space-y-3">
      <div>
        <h1 className="text-2xl">Data Distribution</h1>
        <p className="text-sm text-ink-500 mt-0.5">
          Class-frequency breakdown per hospital. Raw images never leave the node —
          only counts are transmitted.
        </p>
      </div>

      {/* KS / verdict banner */}
      <div className="card flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-3">
          <div>
            <div className="label mb-0">KS statistic</div>
            <div className="text-2xl font-mono text-ink-900">{ks_statistic.toFixed(4)}</div>
          </div>
          <div>
            <div className="label mb-0">Partition verdict</div>
            <span className={`badge-${SKEW_TONE[skew_verdict]}`}>
              <span className={`dot dot-${SKEW_TONE[skew_verdict]}`} />
              {skew_verdict.charAt(0).toUpperCase() + skew_verdict.slice(1)}
            </span>
          </div>
        </div>
        <div className="text-sm text-ink-500 flex-1 min-w-[240px]">
          {SKEW_LABELS[skew_verdict]}
        </div>
      </div>

      {/* Per-node bars */}
      <div className="card space-y-4">
        <h2 className="text-lg">Class distribution by hospital</h2>
        {nodesList.length === 0 ? (
          <p className="text-sm text-ink-500">No nodes registered.</p>
        ) : (
          nodesList.map((n) => {
            const dist = nodeData[n.node_id];
            if (!dist) return null;
            return (
              <NodeBar key={n.node_id} node_id={n.node_id} dist={dist} classes={classes} />
            );
          })
        )}
      </div>

      {/* Per-class totals */}
      <div className="card">
        <h2 className="text-lg mb-2">Global class totals</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-x-4 gap-y-2">
          {classes.map((c) => (
            <div key={c} className="text-sm">
              <div className="font-mono text-xs text-ink-500">{c}</div>
              <div className="font-mono text-base">{(data.global_class_totals[c] ?? 0).toLocaleString()}</div>
            </div>
          ))}
        </div>
        <div className="mt-2 pt-2 border-t border-ink-200 font-mono text-sm">
          Total: {data.global_total_samples.toLocaleString()} images across {classes.length} classes
        </div>
      </div>

      {/* KS by node */}
      <div className="card">
        <h2 className="text-lg mb-2">Per-node KS statistic</h2>
        <table className="table">
          <thead>
            <tr><th>Node</th><th>KS</th><th>Verdict</th></tr>
          </thead>
          <tbody>
            {Object.entries(ks_by_node).map(([nid, ks]) => (
              <tr key={nid}>
                <td className="font-mono">{nid}</td>
                <td className="font-mono">{ks.toFixed(4)}</td>
                <td>
                  <span className={`badge-${SKEW_TONE[skew_verdict]}`}>
                    {skew_verdict.charAt(0).toUpperCase() + skew_verdict.slice(1)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Navigation */}
      <div className="flex gap-2">
        <Link to="/dataset" className="btn-secondary"><Icon.Layers /> Local Dataset table</Link>
        <Link to="/nodes" className="btn-secondary"><Icon.Network /> Node Status</Link>
      </div>
    </div>
  );
}
