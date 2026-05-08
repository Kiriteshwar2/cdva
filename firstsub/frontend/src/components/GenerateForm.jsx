import { useId } from "react";

export default function GenerateForm({
  numSamples,
  onNumSamplesChange,
  selectedCheckpoint,
  onCheckpointChange,
  modelOptions,
  onLoadModel,
  modelLoadState,
  onSubmit,
  loading,
  backendStatus,
}) {
  const inputId = useId();
  const modelReady = backendStatus?.model_loaded;
  const backendOnline = backendStatus?.status === "ok";

  return (
    <div className="form-shell">
      <div className="form-shell__intro">
        <div>
          <p className="form-shell__kicker">Generation Console</p>
          <h2 className="form-shell__title">Generate new crystal candidates</h2>
          <p className="form-shell__description">
            Sample structures from the trained CDVAE, refine them, validate them, and return ready-to-view CIFs.
          </p>
        </div>
        <div className={`status-pill ${modelReady ? "status-pill--success" : "status-pill--warning"}`}>
          <span className="status-pill__dot" />
          {modelReady ? "Model ready" : "Checkpoint missing"}
        </div>
      </div>

      <form
        className="form-grid"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <label className="field" htmlFor={inputId}>
          <span className="field__label">Number of samples</span>
          <input
            id={inputId}
            type="number"
            min="1"
            max="32"
            step="1"
            value={numSamples}
            onChange={(event) => onNumSamplesChange(event.target.value)}
            className="field__input"
          />
        </label>

        <div className="field__stack">
          <label className="field">
            <span className="field__label">Checkpoint / model</span>
            <select
              value={selectedCheckpoint || ""}
              onChange={(event) => onCheckpointChange(event.target.value)}
              className="field__select"
            >
              <option value="">Use active backend model</option>
              {modelOptions.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id}
                </option>
              ))}
            </select>
          </label>
          <div className="field__button-row">
            <button
              type="button"
              onClick={onLoadModel}
              disabled={!selectedCheckpoint || modelLoadState === "loading"}
              className="btn-outline"
            >
              {modelLoadState === "loading" ? "Loading model..." : "Load selected model"}
            </button>
            <button
              type="submit"
              disabled={loading || !modelReady}
              className="btn-primary"
            >
              {loading ? "Generating structures..." : "Generate Structures"}
            </button>
          </div>
          <p className="field__hint">
            {backendOnline ? (
              <>
                Backend: <strong>{backendStatus.device}</strong>
                {backendStatus.model_id ? <> | <strong>{backendStatus.model_id}</strong></> : null}
              </>
            ) : (
              "Set CDVAE_CHECKPOINT_PATH or place a checkpoint in runs/cdvae/checkpoints."
            )}
          </p>
        </div>
      </form>
    </div>
  );
}
