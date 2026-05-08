import { startTransition, useEffect, useMemo, useState } from "react";

import AuthLanding from "../components/AuthLanding";
import GenerationStudio from "../components/GenerationStudio";
import StructureViewer from "../components/StructureViewer";
import ComparisonViewer from "../components/ComparisonViewer";
import SharedViewerToolbar from "../components/SharedViewerToolbar";
import ComparisonMetrics from "../components/ComparisonMetrics";
import ValidationExportRail from "../components/ValidationExportRail";
import { useSession } from "../hooks/useSession";
import { createGeneration, fetchGeneration, fetchHealth, fetchHistory, fetchModels } from "../services/api";

export default function Home() {
  const session = useSession();
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [loading, setLoading] = useState(false);
  const [requestError, setRequestError] = useState("");
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(null);
  const [activePage, setActivePage] = useState("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [historyQuery, setHistoryQuery] = useState("");
  const [favorites, setFavorites] = useState([]);
  const [historySort, setHistorySort] = useState("latest");
  const [settings, setSettings] = useState({
    theme: "scientific-light",
    viewerQuality: "high",
    animations: true,
    defaultExport: "cif",
    distanceUnit: "angstrom",
  });
  const [comparisonGeneration, setComparisonGeneration] = useState(null);
  const [viewerToolbar, setViewerToolbar] = useState({
    showBonds: true,
    showAtoms: true,
    showCell: true,
    autoRotate: false,
  });

  useEffect(() => {
    if (session.status !== "authenticated") {
      return;
    }

    Promise.all([fetchHealth(), fetchModels(), fetchHistory()])
      .then(([healthResponse, modelResponse, historyResponse]) => {
        startTransition(() => {
          setHealth(healthResponse);
          setModels(modelResponse);
          setHistory(historyResponse.items);
          setSelected(historyResponse.items[0] || null);
        });
      })
      .catch((error) => {
        setRequestError(error.response?.data?.detail || error.message || "Failed to load workspace data.");
      });
  }, [session.status]);

  async function handleAuth(action, payload) {
    setAuthBusy(true);
    setAuthError("");
    try {
      await action(payload);
    } catch (error) {
      setAuthError(error.response?.data?.detail || error.message || "Authentication failed.");
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleGenerate(payload) {
    setLoading(true);
    setRequestError("");
    try {
      const generation = await createGeneration(payload);
      const historyResponse = await fetchHistory();
      const healthResponse = await fetchHealth();
      startTransition(() => {
        setSelected(generation);
        setHistory(historyResponse.items);
        setHealth(healthResponse);
      });
    } catch (error) {
      setRequestError(error.response?.data?.detail || error.message || "Generation failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSelect(item) {
    try {
      const detail = await fetchGeneration(item.id);
      setSelected(detail);
    } catch (error) {
      setRequestError(error.response?.data?.detail || error.message || "Failed to load generation.");
    }
  }

  const latest = selected || history[0] || null;
  const recentFeed = useMemo(() => history.slice(0, 10), [history]);
  const dashboardTiles = useMemo(() => history.slice(0, 4), [history]);
  const activeModelName = health?.model_path?.split(/[\\/]/).pop() || models[0]?.checkpoint_name || "Not loaded";
  const filteredHistory = useMemo(() => {
    const normalized = historyQuery.trim().toLowerCase();
    const base = normalized
      ? history.filter((item) => String(item.structure?.formula || "").toLowerCase().includes(normalized))
      : history;
    const sorted = [...base].sort((a, b) => {
      const timeA = new Date(a.created_at).getTime();
      const timeB = new Date(b.created_at).getTime();
      if (historySort === "oldest") return timeA - timeB;
      if (historySort === "atoms") return (b.metadata?.atoms_count || 0) - (a.metadata?.atoms_count || 0);
      return timeB - timeA;
    });
    return sorted;
  }, [history, historyQuery, historySort]);
  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: "▣" },
    { id: "generate", label: "Generate Crystal", icon: "◈" },
    { id: "history", label: "History", icon: "◷" },
    { id: "models", label: "Models", icon: "⌬" },
    { id: "training", label: "Training", icon: "⟲" },
    { id: "settings", label: "Settings", icon: "⚙" },
  ];

  if (session.status !== "authenticated") {
    return (
      <AuthLanding
        busy={authBusy || session.status === "loading"}
        error={authError}
        onLogin={(payload) => handleAuth(session.login, payload)}
        onSignup={(payload) => handleAuth(session.signup, payload)}
      />
    );
  }

  return (
    <div className={`app-shell app-theme--${settings.theme}`}>
      <aside className={`left-sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-brand">
          <strong>CDVAE Lab</strong>
          <small>Materials AI Platform</small>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`sidebar-item ${activePage === item.id ? "is-active" : ""}`}
              onClick={() => {
                setActivePage(item.id);
                setSidebarOpen(false);
              }}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="glass-pill">
            <span>{session.user?.name}</span>
            <small>{session.user?.email}</small>
          </div>
          <button type="button" className="secondary-button sidebar-logout" onClick={session.logout}>
            Logout / Profile
          </button>
        </div>
      </aside>

      <section className="main-area">
        <header className="mobile-topbar">
          <button type="button" className="secondary-button" onClick={() => setSidebarOpen((value) => !value)}>
            {sidebarOpen ? "Close Menu" : "Menu"}
          </button>
        </header>

        {requestError ? <div className="alert-banner">{requestError}</div> : null}

        {activePage === "dashboard" ? (
          <section className="page-block">
            <header className="page-header">
              <div className="eyebrow">Materials AI Workstation</div>
              <h1>Dashboard</h1>
              <p>Overview of system health, active models, generation history, and recent computational activity.</p>
            </header>

            <div className="status-row">
              <article className="status-card status-card--pulse">
                <span className="status-card__label">Backend Online</span>
                <strong>{health?.status || "Loading"}</strong>
                <small>{health?.device || "Detecting device"}</small>
              </article>
              <article className="status-card">
                <span className="status-card__label">Active Model</span>
                <strong>{activeModelName}</strong>
                <small>{health?.model_loaded ? "Singleton active" : "Load pending"}</small>
              </article>
              <article className="status-card">
                <span className="status-card__label">Crystals Generated</span>
                <strong>{history.length}</strong>
                <small>Mongo history entries</small>
              </article>
              <article className="status-card">
                <span className="status-card__label">Activity Logs</span>
                <strong>{recentFeed.length}</strong>
                <small>Tracked recent operations</small>
              </article>
            </div>

            <div className="dashboard-grid">
              <section className="studio-card quick-actions">
                <div className="panel-heading compact">
                  <h3>Quick actions</h3>
                </div>
                <div className="quick-actions__grid">
                  <button type="button" className="secondary-button" onClick={() => setActivePage("generate")}>Generate Crystal</button>
                  {/* <button type="button" className="secondary-button" onClick={() => setActivePage("training")}>Upload Dataset</button>
                  <button type="button" className="secondary-button" onClick={() => setActivePage("training")}>Start Training</button> */}
                  <button type="button" className="secondary-button" onClick={() => setActivePage("history")}>View History</button>
                </div>
              </section>

              <section className="studio-card">
                <div className="panel-heading compact">
                  <h3>Latest generation</h3>
                </div>
                {latest ? (
                  <div className="preview-grid">
                    <div><span>Formula</span><strong>{latest.structure?.formula || "—"}</strong></div>
                    <div><span>Atom count</span><strong>{latest.metadata?.atoms_count ?? "—"}</strong></div>
                    <div><span>Volume</span><strong>{latest.metadata?.volume?.toFixed?.(2) ?? "—"}</strong></div>
                    <div><span>Min distance</span><strong>{latest.metadata?.min_distance?.toFixed?.(2) ?? "—"}</strong></div>
                    <div><span>Validity</span><strong className={latest.metadata?.validity ? "badge-good" : "badge-warn"}>{latest.metadata?.validity ? "Valid" : "Review"}</strong></div>
                  </div>
                ) : (
                  <p className="muted-copy">No generated crystal yet.</p>
                )}
              </section>

              <section className="studio-card activity-feed">
                <div className="panel-heading compact">
                  <h3>Recent activity</h3>
                </div>
                <div className="activity-feed__body">
                  {recentFeed.length === 0 ? (
                    <p className="muted-copy">No activity yet.</p>
                  ) : recentFeed.map((item) => (
                    <button
                      type="button"
                      key={item.id}
                      className="activity-line"
                      onClick={() => {
                        handleSelect(item);
                        setActivePage("generate");
                      }}
                    >
                      <span>{new Date(item.created_at).toLocaleTimeString()}</span>
                      <code>{item.structure?.formula || "Crystal"} | {item.metadata?.atoms_count ?? 0} atoms</code>
                    </button>
                  ))}
                </div>
              </section>

              <section className="studio-card">
                <div className="panel-heading compact">
                  <h3>Recent generations</h3>
                </div>
                <div className="thumb-grid">
                  {dashboardTiles.length === 0 ? (
                    <p className="muted-copy">No preview structures yet.</p>
                  ) : dashboardTiles.map((item) => (
                    <button key={item.id} type="button" className="thumb-card" onClick={() => { handleSelect(item); setActivePage("generate"); }}>
                      <div className="thumb-orb">{item.structure?.formula?.slice(0, 2) || "CR"}</div>
                      <strong>{item.structure?.formula || "Crystal"}</strong>
                      <small>{item.metadata?.atoms_count || 0} atoms</small>
                    </button>
                  ))}
                </div>
              </section>

              <section className="studio-card analytics-row">
                <div className="panel-heading compact">
                  <h3>Generation analytics</h3>
                </div>
                <div className="analytics-grid">
                  <article><span>Avg atoms</span><strong>{history.length ? Math.round(history.reduce((sum, item) => sum + (item.metadata?.atoms_count || 0), 0) / history.length) : 0}</strong></article>
                  <article><span>Valid ratio</span><strong>{history.length ? `${Math.round((history.filter((item) => item.metadata?.validity).length / history.length) * 100)}%` : "0%"}</strong></article>
                  <article><span>Training status</span><strong>{health?.model_loaded ? "Ready" : "Idle"}</strong></article>
                  <article><span>Ops mode</span><strong>Production</strong></article>
                </div>
              </section>
            </div>
          </section>
        ) : null}

        {activePage === "generate" ? (
          <section className="page-block">
            <header className="page-header compact">
              <div className="eyebrow">Crystal Generation Workspace</div>
              <h1>Crystal Structure Analysis</h1>
              <p>Generate, visualize, and compare crystal structures with full analysis metrics and validation tools.</p>
            </header>

            <div className="workflow-container">
              <GenerationStudio
                models={models}
                loading={loading}
                onGenerate={handleGenerate}
                backendHealth={health}
              />

              {/* Dual-panel viewer comparison workspace */}
              <div className="viewer-comparison-grid">
                <div className="primary-viewer">
                  <StructureViewer generation={selected} onLog={console.log} />
                </div>
                <ComparisonViewer
                  generation={comparisonGeneration}
                  onSelectFromHistory={() => {
                    // Could open a modal to select from history
                    alert("Select comparison structure from history");
                  }}
                />
              </div>

              {/* Shared toolbar for viewer controls */}
              <SharedViewerToolbar
                onReset={() => console.log("Reset viewers")}
                onScreenshot={() => console.log("Screenshot")}
                onToggleBonds={() =>
                  setViewerToolbar((t) => ({ ...t, showBonds: !t.showBonds }))
                }
                onToggleAtoms={() =>
                  setViewerToolbar((t) => ({ ...t, showAtoms: !t.showAtoms }))
                }
                onToggleCell={() =>
                  setViewerToolbar((t) => ({ ...t, showCell: !t.showCell }))
                }
                onAutoRotate={() =>
                  setViewerToolbar((t) => ({
                    ...t,
                    autoRotate: !t.autoRotate,
                  }))
                }
                onFullscreen={() => console.log("Fullscreen")}
                showBonds={viewerToolbar.showBonds}
                showAtoms={viewerToolbar.showAtoms}
                showCell={viewerToolbar.showCell}
                autoRotate={viewerToolbar.autoRotate}
              />

              {/* Comparison metrics display */}
              <ComparisonMetrics
                primary={selected?.metadata}
                comparison={comparisonGeneration?.metadata}
              />

              <ValidationExportRail
                generation={selected}
                onLog={console.log}
              />
            </div>
          </section>
        ) : null}

        {activePage === "history" ? (
          <section className="page-block">
            <header className="page-header">
              <div className="eyebrow">Scientific Archive</div>
              <h1>Generation History</h1>
              <p>Searchable vault of all generated structures with validation metadata and quick-access actions.</p>
            </header>
            <section className="studio-card history-archive">
              <div className="history-toolbar">
                <input value={historyQuery} onChange={(event) => setHistoryQuery(event.target.value)} placeholder="Search by formula..." />
                <select value={historySort} onChange={(event) => setHistorySort(event.target.value)}>
                  <option value="latest">Latest</option>
                  <option value="oldest">Oldest</option>
                  <option value="atoms">Atom count</option>
                </select>
              </div>
              <div className="history-archive__grid">
                {filteredHistory.map((item) => {
                  const isFavorite = favorites.includes(item.id);
                  return (
                    <article key={item.id} className="archive-card">
                      <div className="archive-card__preview">{item.structure?.formula?.slice(0, 2) || "CR"}</div>
                      <div>
                        <strong>{item.structure?.formula || "Crystal"}</strong>
                        <small>{new Date(item.created_at).toLocaleString()}</small>
                        <p>Validation: {item.metadata?.validity ? "Valid" : "Review"} | Atoms: {item.metadata?.atoms_count ?? 0} | Density: {item.metadata?.density?.toFixed?.(2) ?? "—"}</p>
                      </div>
                      <div className="archive-card__actions">
                        <button type="button" className="secondary-button" onClick={() => { handleSelect(item); setActivePage("generate"); }}>Open</button>
                        <button type="button" className="secondary-button" onClick={() => setFavorites((current) => isFavorite ? current.filter((id) => id !== item.id) : [...current, item.id])}>
                          {isFavorite ? "Unfavorite" : "Favorite"}
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          </section>
        ) : null}
        {activePage === "models" ? (
          <section className="page-block">
            <header className="page-header">
              <div className="eyebrow">Model Management</div>
              <h1>Models</h1>
              <p>Active checkpoint registry with performance metrics and readiness status.</p>
            </header>
            <section className="models-grid">
              {models.map((model) => {
                const active = model.checkpoint_name === activeModelName;
                return (
                  <article key={model.checkpoint_name} className={`studio-card model-card ${active ? "active" : ""}`}>
                    <div className="panel-heading compact">
                      <h3>{model.checkpoint_name}</h3>
                      <span className={active ? "status-badge is-good" : "status-badge"}>{active ? "Active" : "Available"}</span>
                    </div>
                    <p className="muted-copy">Epochs: {model.epoch ?? "—"} | Source: {model.dataset ?? "mp_20"} | Size: {model.size_mb ? `${model.size_mb} MB` : "—"}</p>
                    <div className="analytics-grid">
                      <article><span>Val acc</span><strong>{model.validation_accuracy ? `${model.validation_accuracy}%` : "—"}</strong></article>
                      <article><span>Status</span><strong>{health?.model_loaded ? "Ready" : "Idle"}</strong></article>
                    </div>
                  </article>
                );
              })}
            </section>
          </section>
        ) : null}
        {activePage === "training" ? (
          <section className="page-block">
            <header className="page-header">
              <div className="eyebrow">Computational Training</div>
              <h1>Training</h1>
              <p>Model development center with dataset management and training progress tracking.</p>
            </header>
            <section className="studio-card">
              <div className="analytics-grid">
                <article><span>Pipeline</span><strong>Ready</strong></article>
                <article><span>Dataset</span><strong>mp_20</strong></article>
                <article><span>Queued jobs</span><strong>0</strong></article>
                <article><span>Last run</span><strong>—</strong></article>
              </div>
            </section>
          </section>
        ) : null}
        {activePage === "settings" ? (
          <section className="page-block">
            <header className="page-header">
              <div className="eyebrow">User Configuration</div>
              <h1>Settings</h1>
              <p>Personalize appearance, viewer preferences, scientific units, and export behavior.</p>
            </header>
            <section className="studio-card settings-grid">
              <label className="field"><span>Theme</span><select value={settings.theme} onChange={(event) => setSettings((state) => ({ ...state, theme: event.target.value }))}><option value="scientific-light">Scientific Light</option><option value="scientific-deep">Scientific Deep</option></select></label>
              <label className="field"><span>Viewer quality</span><select value={settings.viewerQuality} onChange={(event) => setSettings((state) => ({ ...state, viewerQuality: event.target.value }))}><option value="high">High</option><option value="balanced">Balanced</option><option value="performance">Performance</option></select></label>
              <label className="field"><span>Default export</span><select value={settings.defaultExport} onChange={(event) => setSettings((state) => ({ ...state, defaultExport: event.target.value }))}><option value="cif">CIF</option><option value="json">JSON</option><option value="both">CIF + JSON</option></select></label>
              <label className="field"><span>Distance unit</span><select value={settings.distanceUnit} onChange={(event) => setSettings((state) => ({ ...state, distanceUnit: event.target.value }))}><option value="angstrom">Angstrom</option><option value="nm">Nanometer</option></select></label>
              <label className="field checkbox-field"><span>Animations</span><input type="checkbox" checked={settings.animations} onChange={(event) => setSettings((state) => ({ ...state, animations: event.target.checked }))} /></label>
            </section>
          </section>
        ) : null}
      </section>
    </div>
  );
}
