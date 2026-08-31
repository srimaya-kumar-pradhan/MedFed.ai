export function HelpPage() {
  return (
    <div className="p-4 max-w-3xl space-y-3">
      <div>
        <h1 className="text-2xl">Help</h1>
        <p className="text-sm text-ink-500 mt-0.5">Quick reference for the Clinical Portal.</p>
      </div>

      <div className="card">
        <h2 className="text-lg mb-2">Analyzing an image</h2>
        <ol className="list-decimal list-inside text-sm text-ink-700 space-y-1.5 leading-relaxed">
          <li>Open the Analyze Image page from the sidebar.</li>
          <li>Drop a chest X-ray (PNG or JPEG) into the upload area, or click to browse.</li>
          <li>Click "Analyze Image" to send it to the inference service.</li>
          <li>Review the AI assessment, top predictions, and the Grad-CAM overlay.</li>
          <li>Use the visualization as a decision support input. Final clinical judgement remains with you.</li>
        </ol>
      </div>

      <div className="card">
        <h2 className="text-lg mb-2">Limitations</h2>
        <ul className="list-disc list-inside text-sm text-ink-700 space-y-1.5 leading-relaxed">
          <li>Model is trained on the NIH Chest X-ray prototype dataset (14-class multi-label).</li>
          <li>Predictions are AI-assisted and do not replace clinical judgement or formal radiology review.</li>
          <li>Model is not FDA-cleared, CE-marked, or otherwise approved for clinical use.</li>
          <li>Outputs are intended for research, demonstration, and product evaluation only.</li>
        </ul>
      </div>

      <div className="card">
        <h2 className="text-lg mb-2">Privacy</h2>
        <p className="text-sm text-ink-700 leading-relaxed">
          Your image is sent to the inference service running in this environment. It is not stored on the
          server. The model is trained collaboratively across hospitals without raw patient data ever leaving
          each hospital's infrastructure; only model parameters cross the network.
        </p>
      </div>
    </div>
  );
}
