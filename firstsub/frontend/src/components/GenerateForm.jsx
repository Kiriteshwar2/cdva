import { useId } from "react";

export default function GenerateForm({
  numSamples,
  onNumSamplesChange,
  onSubmit,
  loading,
  backendStatus,
}) {
  const inputId = useId();
  const modelReady = backendStatus?.model_loaded;

  return (
    <section className="rounded-[28px] border border-slate-200/80 bg-white/85 p-6 shadow-[0_24px_80px_rgba(15,23,42,0.12)] backdrop-blur">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-700">
            Generation Console
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-900">Generate new crystal candidates</h2>
          <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
            Sample structures from the trained CDVAE, refine them, validate them, and return ready-to-view CIFs.
          </p>
        </div>
        <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${
          modelReady
            ? "bg-emerald-100 text-emerald-700"
            : "bg-amber-100 text-amber-700"
        }`}>
          <span className={`h-2 w-2 rounded-full ${modelReady ? "bg-emerald-500" : "bg-amber-500"}`} />
          {modelReady ? "Model ready" : "Checkpoint missing"}
        </div>
      </div>

      <form
        className="grid gap-4 md:grid-cols-[minmax(0,220px)_1fr]"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <label className="flex flex-col gap-2" htmlFor={inputId}>
          <span className="text-sm font-medium text-slate-700">Number of samples</span>
          <input
            id={inputId}
            type="number"
            min="1"
            max="32"
            step="1"
            value={numSamples}
            onChange={(event) => onNumSamplesChange(event.target.value)}
            className="h-12 rounded-2xl border border-slate-200 bg-slate-50 px-4 text-base text-slate-900 outline-none transition focus:border-cyan-500 focus:bg-white"
          />
        </label>

        <div className="flex flex-col justify-end gap-3">
          <button
            type="submit"
            disabled={loading || !modelReady}
            className="inline-flex h-12 items-center justify-center rounded-2xl bg-slate-950 px-5 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? "Generating structures..." : "Generate Structures"}
          </button>
          <p className="text-sm text-slate-500">
            {modelReady
              ? `Backend: ${backendStatus.device}${backendStatus.checkpoint_path ? ` | ${backendStatus.checkpoint_path}` : ""}`
              : "Set CDVAE_CHECKPOINT_PATH or place a checkpoint in runs/cdvae/checkpoints."}
          </p>
        </div>
      </form>
    </section>
  );
}
