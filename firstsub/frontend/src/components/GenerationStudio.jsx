import { useEffect, useMemo, useState } from "react";

const ELEMENT_CATEGORIES = {
  "Core Non-metals": ["H", "C", "N", "O", "F", "P", "S", "Cl"],
  Semiconductors: ["Si", "Ge", "Ga", "In", "As", "Se", "Te"],
  "Transition Metals": ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mo", "W"],
  "Battery Materials": ["Li", "Na", "K", "Mg", "Al", "Ni", "Co", "Mn", "Fe", "P"],
  Lanthanides: ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy"],
  Actinides: ["Th", "U", "Np", "Pu"],
};
const GENERATION_STEPS = [
  "Initializing latent diffusion...",
  "Sampling crystal candidates...",
  "Optimizing lattice geometry...",
  "Validating interatomic distances...",
  "Computing structural metrics...",
  "Exporting valid structure...",
];

function NumberField({ label, name, value, onChange, step = "0.1", min, placeholder }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        name={name}
        type="number"
        value={value}
        step={step}
        min={min}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function AccordionSection({ id, title, openId, setOpenId, children }) {
  const isOpen = openId === id;
  return (
    <section className="accordion">
      <button type="button" className={`accordion__head ${isOpen ? "open" : ""}`} onClick={() => setOpenId(isOpen ? "" : id)}>
        <strong>{title}</strong>
        <span>{isOpen ? "−" : "+"}</span>
      </button>
      <div className={`accordion__body ${isOpen ? "open" : ""}`}>{children}</div>
    </section>
  );
}

export default function GenerationStudio({ models, loading, onGenerate, backendHealth }) {
  const [openId, setOpenId] = useState("basic");
  const [elementQuery, setElementQuery] = useState("");
  const [stepIndex, setStepIndex] = useState(0);
  const [form, setForm] = useState({
    checkpoint_name: models[0]?.checkpoint_name || "",
    elements: ["C", "Si", "O"],
    num_atoms: "12",
    lattice: { a: "7.2", b: "7.2", c: "7.2", alpha: "90", beta: "90", gamma: "90" },
    target_properties: { energy_min: "-1.5", energy_max: "1.0", density_min: "1.5", density_max: "5.0" },
    min_interatomic_distance: "1.4",
    candidate_pool_size: "24",
    max_attempts: "64",
  });

  const selectedCheckpoint = form.checkpoint_name || models[0]?.checkpoint_name || "";
  const filteredCategories = useMemo(() => {
    const query = elementQuery.trim().toLowerCase();
    if (!query) return ELEMENT_CATEGORIES;
    return Object.fromEntries(
      Object.entries(ELEMENT_CATEGORIES).map(([category, symbols]) => [
        category,
        symbols.filter((symbol) => symbol.toLowerCase().includes(query)),
      ]).filter(([, symbols]) => symbols.length > 0),
    );
  }, [elementQuery]);

  useEffect(() => {
    if (!loading) {
      setStepIndex(0);
      return undefined;
    }
    const interval = window.setInterval(() => {
      setStepIndex((current) => (current + 1) % GENERATION_STEPS.length);
    }, 1300);
    return () => window.clearInterval(interval);
  }, [loading]);

  const toggleElement = (element) => {
    setForm((current) => ({
      ...current,
      elements: current.elements.includes(element)
        ? current.elements.filter((item) => item !== element)
        : [...current.elements, element],
    }));
  };

  const updateTopLevel = (field, value) => {
    setForm((current) => ({ ...current, [field]: value }));
  };

  const updateNested = (group, field, value) => {
    setForm((current) => ({
      ...current,
      [group]: {
        ...current[group],
        [field]: value,
      },
    }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onGenerate({
      checkpoint_name: selectedCheckpoint || undefined,
      elements: form.elements,
      num_atoms: Number(form.num_atoms),
      lattice: {
        a: Number(form.lattice.a),
        b: Number(form.lattice.b),
        c: Number(form.lattice.c),
        alpha: Number(form.lattice.alpha),
        beta: Number(form.lattice.beta),
        gamma: Number(form.lattice.gamma),
      },
      target_properties: {
        energy_min: Number(form.target_properties.energy_min),
        energy_max: Number(form.target_properties.energy_max),
        density_min: Number(form.target_properties.density_min),
        density_max: Number(form.target_properties.density_max),
      },
      min_interatomic_distance: Number(form.min_interatomic_distance),
      candidate_pool_size: Number(form.candidate_pool_size),
      max_attempts: Number(form.max_attempts),
    });
  };

  return (
    <form className="studio-card generation-controls" onSubmit={handleSubmit}>
      {loading ? (
        <div className="generation-overlay">
          <div className="generation-overlay__wireframe">◈</div>
          <p>Generating crystal structure...</p>
          <small>{GENERATION_STEPS[stepIndex]}</small>
        </div>
      ) : null}
      <div className="panel-heading">
        <div>
          <h3>Generation</h3>
          <small style={{ color: "var(--muted)", display: "block", marginTop: "4px", fontSize: "0.85rem" }}>Configure constraints</small>
        </div>
      </div>

      <fieldset disabled={loading} className="generation-fieldset">
      <AccordionSection id="basic" title="Basic" openId={openId} setOpenId={setOpenId}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "8px" }}>
          <label className="field">
            <span>Checkpoint</span>
            <select value={selectedCheckpoint} onChange={(event) => updateTopLevel("checkpoint_name", event.target.value)}>
              {models.map((model) => (
                <option key={model.checkpoint_name} value={model.checkpoint_name}>
                  {model.checkpoint_name}
                </option>
              ))}
            </select>
          </label>
          <NumberField label="Atoms" name="num_atoms" min="1" step="1" placeholder="12" value={form.num_atoms} onChange={(value) => updateTopLevel("num_atoms", value)} />
        </div>
        <div className="chip-picker">
          <div className="chip-picker__header">
            <span className="field__label">Elements</span>
            <input
              className="element-search"
              placeholder="Search..."
              value={elementQuery}
              onChange={(event) => setElementQuery(event.target.value)}
            />
          </div>
          <div className="element-categories">
            {Object.entries(filteredCategories).map(([category, symbols]) => (
              <section key={category} className="element-category">
                <h5>{category}</h5>
                <div className="chip-picker__grid">
                  {symbols.map((element) => (
                    <label key={element} className="chip-option">
                      <input type="checkbox" checked={form.elements.includes(element)} onChange={() => toggleElement(element)} />
                      <span>{element}</span>
                    </label>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </div>
      </AccordionSection>

      <AccordionSection id="lattice" title="Lattice" openId={openId} setOpenId={setOpenId}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
          <NumberField label="a" name="lattice_a" value={form.lattice.a} onChange={(value) => updateNested("lattice", "a", value)} />
          <NumberField label="b" name="lattice_b" value={form.lattice.b} onChange={(value) => updateNested("lattice", "b", value)} />
          <NumberField label="c" name="lattice_c" value={form.lattice.c} onChange={(value) => updateNested("lattice", "c", value)} />
          <NumberField label="α" name="alpha" value={form.lattice.alpha} onChange={(value) => updateNested("lattice", "alpha", value)} />
          <NumberField label="β" name="beta" value={form.lattice.beta} onChange={(value) => updateNested("lattice", "beta", value)} />
          <NumberField label="γ" name="gamma" value={form.lattice.gamma} onChange={(value) => updateNested("lattice", "gamma", value)} />
        </div>
      </AccordionSection>

      <AccordionSection id="advanced" title="Constraints" openId={openId} setOpenId={setOpenId}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "8px" }}>
          <NumberField label="Energy min" name="energy_min" value={form.target_properties.energy_min} onChange={(value) => updateNested("target_properties", "energy_min", value)} />
          <NumberField label="Energy max" name="energy_max" value={form.target_properties.energy_max} onChange={(value) => updateNested("target_properties", "energy_max", value)} />
          <NumberField label="Density min" name="density_min" value={form.target_properties.density_min} onChange={(value) => updateNested("target_properties", "density_min", value)} />
          <NumberField label="Density max" name="density_max" value={form.target_properties.density_max} onChange={(value) => updateNested("target_properties", "density_max", value)} />
          <NumberField label="Min distance" name="min_interatomic_distance" value={form.min_interatomic_distance} onChange={(value) => updateTopLevel("min_interatomic_distance", value)} />
          <NumberField label="Pool size" name="candidate_pool_size" value={form.candidate_pool_size} step="1" onChange={(value) => updateTopLevel("candidate_pool_size", value)} />
          <NumberField label="Max attempts" name="max_attempts" value={form.max_attempts} step="1" onChange={(value) => updateTopLevel("max_attempts", value)} />
        </div>
      </AccordionSection>

      <button className="primary-button" type="submit" disabled={loading} style={{ marginTop: "8px", width: "100%" }}>
        {loading ? "Generating..." : "Generate"}
      </button>
      </fieldset>
    </form>
  );
}
