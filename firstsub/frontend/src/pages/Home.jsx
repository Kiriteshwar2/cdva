import { useEffect, useMemo, useState } from "react";

import GenerateForm from "../components/GenerateForm";
import ResultsGrid from "../components/ResultsGrid";
import { API_BASE_URL, fetchHealth, generateStructures } from "../services/api";

export default function Home() {
  const [numSamples, setNumSamples] = useState(4);
  const [structures, setStructures] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendStatus, setBackendStatus] = useState(null);

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((status) => {
        if (active) {
          setBackendStatus(status);
        }
      })
      .catch((requestError) => {
        if (active) {
          setError(requestError.response?.data?.detail || requestError.message);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  async function handleGenerate() {
    setLoading(true);
    setError("");
    try {
      const response = await generateStructures(Number(numSamples));
      setStructures(response.structures);
      const latestHealth = await fetchHealth();
      setBackendStatus(latestHealth);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message || "Generation failed.");
    } finally {
      setLoading(false);
    }
  }

  const validCount = useMemo(
    () => structures.filter((structure) => structure.valid).length,
    [structures],
  );

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(8,145,178,0.18),transparent_32%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_100%)]">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8 overflow-hidden rounded-[32px] border border-white/70 bg-[linear-gradient(135deg,#082f49_0%,#0f172a_55%,#164e63_100%)] px-6 py-8 text-white shadow-[0_30px_100px_rgba(8,47,73,0.35)] sm:px-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="text-xs font-semibold uppercase tracking-[0.32em] text-cyan-200">Crystal Generator (CDVAE)</p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
                Research-grade crystal generation in a production-style app
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-200 sm:text-base">
                Generate candidate crystal structures with the CDVAE backend, inspect validation metadata, render CIFs in 3D, and download outputs directly from a clean dashboard.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-2xl bg-white/10 px-4 py-4 backdrop-blur">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-300">Backend</p>
                <p className="mt-2 text-xl font-semibold">{backendStatus?.status || "..."}</p>
              </div>
              <div className="rounded-2xl bg-white/10 px-4 py-4 backdrop-blur">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-300">Model</p>
                <p className="mt-2 text-xl font-semibold">{backendStatus?.model_loaded ? "Loaded" : "Missing"}</p>
              </div>
              <div className="rounded-2xl bg-white/10 px-4 py-4 backdrop-blur">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-300">Valid Results</p>
                <p className="mt-2 text-xl font-semibold">{structures.length ? `${validCount}/${structures.length}` : "0/0"}</p>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 space-y-8">
          <GenerateForm
            numSamples={numSamples}
            onNumSamplesChange={setNumSamples}
            onSubmit={handleGenerate}
            loading={loading}
            backendStatus={backendStatus}
          />

          {loading ? (
            <section className="rounded-[28px] border border-slate-200 bg-white/80 p-10 text-center shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
              <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-slate-200 border-t-cyan-600" />
              <h2 className="mt-5 text-2xl font-semibold text-slate-900">Sampling structures...</h2>
              <p className="mt-3 text-sm text-slate-600">
                The backend is running latent sampling, refinement, validation, and CIF conversion.
              </p>
            </section>
          ) : null}

          {error ? (
            <section className="rounded-[24px] border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-700 shadow-sm">
              <span className="font-semibold">Request failed:</span> {error}
            </section>
          ) : null}

          <ResultsGrid structures={structures} />
        </main>

        <footer className="mt-10 flex flex-col gap-2 border-t border-slate-200/80 pt-6 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <p>Frontend talks to <span className="font-medium text-slate-700">{API_BASE_URL}</span></p>
          <p>Flow: generate → refine → validate → inspect → download</p>
        </footer>
      </div>
    </div>
  );
}
