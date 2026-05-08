export default function HistoryRail({ items, selectedId, onSelect }) {
  return (
    <aside className="history-rail">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">History</div>
          <h3>Persisted generations</h3>
        </div>
      </div>

      <div className="history-list">
        {items.length === 0 ? (
          <p className="muted-copy">No generations yet. Your first accepted CIF will appear here.</p>
        ) : (
          items.map((item) => (
            <button
              key={item.id}
              type="button"
              className={selectedId === item.id ? "history-item is-active" : "history-item"}
              onClick={() => onSelect(item)}
            >
              <div>
                <strong>{item.structure.formula}</strong>
                <span>{new Date(item.created_at).toLocaleString()}</span>
              </div>
              <small>{item.metadata.atoms_count} atoms</small>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}
