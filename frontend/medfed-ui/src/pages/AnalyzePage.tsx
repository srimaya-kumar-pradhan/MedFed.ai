import { useEffect, useRef, useState } from "react";
import { predict } from "../api/client";
import type { Prediction, Explanation } from "../api/client";
import { Icon } from "../components/Icon";

type Stage = "upload" | "preview" | "analyzing" | "result";

export function AnalyzePage() {
  const [stage, setStage] = useState<Stage>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  const onPick = (f: File | null | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("File must be an image (PNG or JPEG).");
      return;
    }
    setError(null);
    setPrediction(null);
    setExplanation(null);
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
    setStage("preview");
  };

  const analyze = async () => {
    if (!file) return;
    setError(null);
    setStage("analyzing");
    try {
      const [pred, expl] = await Promise.all([predict.predict(file), predict.explain(file)]);
      setPrediction(pred);
      setExplanation(expl);
      setStage("result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setStage("preview");
    }
  };

  const reexplain = async (targetClass: string) => {
    if (!file) return;
    try {
      const expl = await predict.explain(file, targetClass);
      setExplanation(expl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh explanation");
    }
  };

  const reset = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setPrediction(null);
    setExplanation(null);
    setError(null);
    setStage("upload");
  };

  return (
    <div className="p-4 max-w-6xl space-y-3">
      <div>
        <h1 className="text-2xl">Analyze Image</h1>
        <p className="text-sm text-ink-500 mt-0.5">
          Upload a chest X-ray to receive an AI-assisted prediction. Clinical decision remains with the qualified healthcare professional.
        </p>
      </div>

      {error && (
        <div className="border border-status-bad text-status-bad text-sm px-3 py-2 rounded-sm">
          {error}
        </div>
      )}

      {/* Stage 1: Upload */}
      {stage === "upload" && (
        <div
          className={`card border-dashed ${dragOver ? "border-ink-900 bg-ink-50" : "border-ink-300"} cursor-pointer`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            onPick(e.dataTransfer.files[0]);
          }}
        >
          <div className="py-10 text-center">
            <div className="mx-auto w-10 h-10 text-ink-400 mb-2 flex items-center justify-center border border-ink-200 rounded-sm">
              <Icon.Upload />
            </div>
            <div className="text-base">Drop a chest X-ray here or click to browse</div>
            <div className="text-xs text-ink-500 mt-1">PNG or JPEG · 64×64 minimum · 224×224 recommended</div>
          </div>
          <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => onPick(e.target.files?.[0])} />
        </div>
      )}

      {/* Stage 2: Preview + Analyze button */}
      {stage !== "upload" && (
        <div className="card">
          <div className="grid grid-cols-1 md:grid-cols-[1fr_320px] gap-3">
            <div className="border border-ink-200 bg-ink-50 p-2 flex items-center justify-center min-h-[260px]">
              {previewUrl && <img src={previewUrl} alt="Uploaded" className="max-h-[420px] max-w-full object-contain" />}
            </div>
            <div>
              <div className="label">File</div>
              <div className="text-sm font-mono text-ink-900 break-all">{file?.name ?? "—"}</div>
              <div className="mt-2 label">Size</div>
              <div className="text-sm font-mono">{file ? `${(file.size / 1024).toFixed(1)} KB` : "—"}</div>
              <div className="mt-2 label">Type</div>
              <div className="text-sm font-mono">{file?.type ?? "—"}</div>

              {stage === "preview" && (
                <div className="mt-3 flex flex-col gap-2">
                  <button className="btn-primary" onClick={analyze}><Icon.Play /> Analyze Image</button>
                  <button className="btn-secondary" onClick={reset}>Choose a different image</button>
                </div>
              )}
              {stage === "analyzing" && (
                <div className="mt-3 flex flex-col gap-2">
                  <div className="border border-ink-200 px-2 py-2 text-sm flex items-center gap-2">
                    <Icon.Refresh />
                    <span>Running inference…</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Stage 3: Results */}
      {stage === "result" && prediction && explanation && (
        <div className="space-y-3">
          {/* AI Assessment panel — highest model score, never a diagnosis */}
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg">AI Assessment</h2>
              <span className="badge-neutral">AI-assisted prediction</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <div className="label">Highest model score</div>
                <div className="text-2xl mt-1">{explanation.predicted_label}</div>
                <div className="mt-1 text-sm text-ink-500">
                  Confidence: <span className="font-mono text-ink-900">{(explanation.predicted_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="mt-2 inline-block h-1.5 w-full bg-ink-100 rounded-sm overflow-hidden">
                  <div className="h-full bg-accent" style={{ width: `${Math.min(100, explanation.predicted_confidence * 100)}%` }} />
                </div>
                <p className="text-xs text-ink-500 mt-3 leading-relaxed">
                  This is the highest model score, not a diagnosis. The output is generated by a federated-trained model and is intended to support clinical review. Clinical decision remains with the qualified healthcare professional.
                </p>
              </div>

              <div>
                <div className="label">Model predictions</div>
                <table className="table mt-1">
                  <thead>
                    <tr>
                      <th>Finding</th>
                      <th className="text-right">Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prediction.all_predictions
                      .slice()
                      .sort((a, b) => b.probability - a.probability)
                      .map((p) => (
                        <tr key={p.label}>
                          <td>{p.label}</td>
                          <td className="font-mono text-right">{(p.probability * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
                <div className="mt-2 text-xs text-ink-500">
                  Highest model score: <span className="font-medium text-ink-900">{explanation.predicted_label}</span> — {(explanation.predicted_confidence * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>

          {/* Grad-CAM panel — class selector + explained-class label */}
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg">Visual Explanation (Grad-CAM)</h2>
              <span className="badge-neutral">Model interpretability</span>
            </div>

            <div className="flex items-center gap-2 mb-3">
              <label htmlFor="explain-class" className="label">Explain prediction:</label>
              <select
                id="explain-class"
                className="input max-w-[280px]"
                value={explanation.explained_class}
                onChange={(e) => reexplain(e.target.value)}
              >
                {prediction.all_predictions
                  .slice()
                  .sort((a, b) => b.probability - a.probability)
                  .map((p) => (
                    <option key={p.label} value={p.label}>
                      {p.label} — {(p.probability * 100).toFixed(1)}%
                    </option>
                  ))}
              </select>
              <span className="text-xs text-ink-500">
                Model confidence for selected class: <span className="font-mono text-ink-900">{(explanation.explained_class_confidence * 100).toFixed(1)}%</span>
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <div className="label mb-1">Original</div>
                {previewUrl && <img src={previewUrl} alt="Original" className="border border-ink-200 w-full object-contain" />}
              </div>
              <div>
                <div className="label mb-1">AI Explanation</div>
                <div className="border border-ink-200 p-2 text-xs text-ink-700 leading-relaxed">
                  <div className="text-sm font-medium text-ink-900 mb-1">
                    Predicted finding: <span className="font-mono">{explanation.explained_class}</span>
                  </div>
                  <div>
                    Confidence: <span className="font-mono">{(explanation.explained_class_confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-1.5">
                    Interpretation: Model attention is concentrated in the region contributing most strongly to this prediction.
                  </div>
                  <div className="mt-1.5 italic text-ink-500">
                    This is an AI-generated explanation and does not constitute a clinical diagnosis.
                  </div>
                </div>
              </div>
              <div>
                <div className="label mb-1">Grad-CAM overlay</div>
                {explanation.gradcam_png_base64 ? (
                  <img
                    src={`data:image/png;base64,${explanation.gradcam_png_base64}`}
                    alt={`Grad-CAM overlay for ${explanation.explained_class}`}
                    className="border border-ink-200 w-full object-contain"
                  />
                ) : (
                  <div className="border border-ink-200 h-32 flex items-center justify-center text-xs text-ink-500">
                    No overlay available
                  </div>
                )}
                <div className="text-xs text-ink-500 mt-1">
                  Grad-CAM for: <span className="font-medium text-ink-900">{explanation.explained_class}</span>
                </div>
              </div>
            </div>
            <p className="text-xs text-ink-500 mt-2 leading-relaxed">{explanation.caption}</p>
            <p className="text-xs text-ink-500 mt-1 leading-relaxed">
              {explanation.disclaimer}
            </p>
          </div>

          <div className="flex gap-2">
            <button className="btn-primary" onClick={reset}><Icon.Image /> Analyze another image</button>
          </div>
        </div>
      )}
    </div>
  );
}
