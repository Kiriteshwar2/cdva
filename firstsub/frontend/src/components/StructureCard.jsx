import CrystalViewer from "./CrystalViewer";

function downloadCif(structure) {
  const blob = new Blob([structure.cif_string], { type: "chemical/x-cif" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${structure.id}.cif`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function StructureCard({ structure }) {
  const { metadata } = structure;

  return (
    <article className="overflow-hidden rounded-[28px] border border-slate-200/80 bg-white shadow-[0_20px_70px_rgba(15,23,42,0.10)]">
      <div className="border-b border-slate-200/80 bg-[linear-gradient(135deg,rgba(8,145,178,0.10),rgba(15,23,42,0.02))] px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Structure</p>
            <h3 className="mt-2 text-lg font-semibold text-slate-950">{structure.id}</h3>
            <p className="mt-1 text-sm text-slate-600">{metadata.formula || "Unknown formula"}</p>
          </div>
          <span className={`rounded-full px-3 py-1 text-xs font-semibold ${
            structure.valid ? "bg-emerald-100 text-emerald-700" : "bg-rose-100 text-rose-700"
          }`}>
            {structure.valid ? "Valid" : "Needs review"}
          </span>
        </div>
      </div>

      <div className="grid gap-5 p-5 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-2xl bg-slate-50 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Atoms</p>
              <p className="mt-2 text-2xl font-semibold text-slate-900">{metadata.num_atoms}</p>
            </div>
            <div className="rounded-2xl bg-slate-50 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Volume</p>
              <p className="mt-2 text-2xl font-semibold text-slate-900">{metadata.volume.toFixed(2)}</p>
            </div>
            <div className="rounded-2xl bg-slate-50 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Min Distance</p>
              <p className="mt-2 text-2xl font-semibold text-slate-900">
                {Number.isFinite(metadata.minimum_pair_distance)
                  ? metadata.minimum_pair_distance.toFixed(2)
                  : "n/a"}
              </p>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Atom types</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {structure.atom_types.map((atomType, index) => (
                <span key={`${structure.id}-${atomType}-${index}`} className="rounded-full bg-cyan-50 px-3 py-1 text-sm font-medium text-cyan-700">
                  {atomType}
                </span>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 p-4">
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">CIF export</p>
              <button
                type="button"
                onClick={() => downloadCif(structure)}
                className="rounded-xl bg-slate-950 px-3 py-2 text-sm font-semibold text-white transition hover:bg-slate-800"
              >
                Download CIF
              </button>
            </div>
            <pre className="mt-3 max-h-52 overflow-auto rounded-2xl bg-slate-950 p-4 text-xs leading-6 text-slate-100">
              {structure.cif_string}
            </pre>
          </div>
        </div>

        <CrystalViewer cifString={structure.cif_string} structureId={structure.id} />
      </div>
    </article>
  );
}
