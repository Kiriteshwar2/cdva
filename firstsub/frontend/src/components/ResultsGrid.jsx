import StructureCard from "./StructureCard";

export default function ResultsGrid({ structures }) {
  if (!structures.length) {
    return (
      <div className="rounded-[28px] border border-dashed border-slate-300 bg-white/60 p-10 text-center shadow-[0_20px_70px_rgba(15,23,42,0.08)]">
        <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-500">Results</p>
        <h2 className="mt-3 text-2xl font-semibold text-slate-900">No structures generated yet</h2>
        <p className="mx-auto mt-3 max-w-2xl text-sm leading-6 text-slate-600">
          Request one or more samples and the generated crystal cards will appear here with validation details, CIF exports, and 3D previews.
        </p>
      </div>
    );
  }

  return (
    <section className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-700">Results</p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-950">Generated structures</h2>
        </div>
        <p className="text-sm text-slate-500">{structures.length} returned from backend</p>
      </div>
      <div className="grid gap-6">
        {structures.map((structure) => (
          <StructureCard key={structure.id} structure={structure} />
        ))}
      </div>
    </section>
  );
}
