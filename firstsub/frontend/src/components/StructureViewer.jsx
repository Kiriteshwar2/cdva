import StructureViewer3D from "./StructureViewer3D";

export default function StructureViewer({ generation }) {
  if (!generation || !generation.metadata) {
    return (
      <article className="structure-viewer">
        <div className="structure-viewer__empty">
          <div className="structure-viewer__icon">⬡</div>
          <h3>No structure generated</h3>
          <p>
            Configure generation constraints and create a crystal structure
          </p>
        </div>
      </article>
    );
  }

  const { metadata, output_cif, structure } = generation;

  return (
    <article className="structure-viewer">
      <header className="structure-viewer__header">
        <div>
          <h2>{structure?.formula || "Crystal"}</h2>
          <p>
            {metadata.atoms_count} atoms {" • "} {metadata.space_group || "Unknown symmetry"}
          </p>
        </div>
        <div className={`structure-viewer__badge ${metadata.validity ? "valid" : "review"}`}>
          {metadata.validity ? "Valid" : "Review"}
        </div>
      </header>

      <div className="structure-viewer__content">
        <StructureViewer3D
          cifString={output_cif || ""}
          formula={structure?.formula || "Crystal"}
        />
      </div>
    </article>
  );
}