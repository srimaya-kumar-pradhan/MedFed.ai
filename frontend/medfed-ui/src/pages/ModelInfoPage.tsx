import { useEffect, useState } from "react";
import { system } from "../api/client";

type ModelStatus = Awaited<ReturnType<typeof system.modelStatus>>;

export function ModelInfoPage() {
  const [status, setStatus] = useState<ModelStatus | null>(null);

  useEffect(() => { system.modelStatus().then(setStatus).catch(() => undefined); }, []);

  return (
    <div className="p-4 max-w-4xl space-y-3">
      <div>
        <h1 className="text-2xl">Model Information</h1>
        <p className="text-sm text-ink-500 mt-0.5">What model is currently serving inference and where it came from.</p>
      </div>

      {status?.available ? (
        <div className="card">
          <table className="table">
            <tbody>
              <Row k="Status" v={status.registry.status ?? "—"} />
              <Row k="Version" v={status.registry.current_version ?? "—"} mono />
              <Row k="Path" v={status.registry.path ?? "—"} mono small />
              <Row k="Architecture" v={String(status.registry.metadata?.architecture ?? "—")} mono />
              <Row k="Number of classes" v={String(status.registry.metadata?.num_classes ?? "—")} mono />
              <Row k="Study type" v={String(status.registry.metadata?.study_type ?? "—")} />
              <Row k="Task" v={String(status.registry.metadata?.task ?? "—")} />
              <Row k="Dataset" v={String(status.registry.metadata?.dataset ?? "—")} />
              <Row k="Training round" v={status.registry.round ? String(status.registry.round) : "N/A"} mono />
              <Row k="Strategy" v={String(status.registry.metadata?.strategy ?? "—")} mono />
              <Row k="Privacy" v={String(status.registry.metadata?.privacy ?? "—")} mono />
              <Row k="Macro F1" v={fmt(status.registry.metrics?.f1)} mono />
              <Row k="ROC-AUC" v={fmt(status.registry.metrics?.roc_auc)} mono />
              <Row k="Loss" v={fmt(status.registry.metrics?.loss)} mono />
              <Row k="Device" v={status.device} mono />
              <Row k="Loaded for" v={`${status.loaded_for_seconds}s`} mono />
              <Row k="Created" v={status.registry.created_at ?? "—"} mono small />
            </tbody>
          </table>
        </div>
      ) : (
        <div className="card">
          <p className="text-sm text-ink-500">Model is currently unavailable. Please contact the system administrator.</p>
        </div>
      )}

      <div className="card">
        <h2 className="text-lg mb-2">About this model</h2>
        <p className="text-sm text-ink-500 leading-relaxed">
          This model is a DenseNet121 backbone fine-tuned for multi-label chest X-ray classification across 14 pathology classes.
          It is part of a federated training pipeline in which each hospital trains on its own local data; only model parameter updates
          cross the network. The model is intended for prototype research and demonstration of the federated infrastructure.
          It is not a clinically validated medical device. Clinical decision remains with the qualified healthcare professional.
        </p>
      </div>
    </div>
  );
}

function fmt(v: number | null | undefined) {
  return v === null || v === undefined ? "N/A" : v.toFixed(4);
}

function Row({ k, v, mono, small }: { k: string; v: string; mono?: boolean; small?: boolean }) {
  return (
    <tr>
      <td className="text-ink-500 w-1/3">{k}</td>
      <td className={`${mono ? "font-mono" : ""} ${small ? "text-xs" : ""} break-all`}>{v}</td>
    </tr>
  );
}
