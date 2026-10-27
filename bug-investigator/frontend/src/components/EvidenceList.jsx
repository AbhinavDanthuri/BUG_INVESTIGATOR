import { useState } from "react";

export default function EvidenceList({ evidence }) {
  const [showAll, setShowAll] = useState(false);
  if (!evidence.length) return null;

  const cited = evidence.filter((e) => e.cited);
  const shown = showAll ? evidence : cited.length ? cited : evidence.slice(0, 3);

  return (
    <section className="panel">
      <h2>
        Evidence — {cited.length} cited of {evidence.length} retrieved
      </h2>
      {shown.map((e, i) => (
        <div className="evidence-item" key={i}>
          <div className="step-head">
            <span className="tag">{e.source_type}</span>
            <span className="mono">
              {e.ref}
              {e.start_line ? `:${e.start_line}-${e.end_line}` : ""}
            </span>
            {e.cited && <span className="cited">cited</span>}
            <span className="muted">score {e.score.toFixed(4)}</span>
          </div>
          <pre className="block">{e.text.slice(0, 1200)}</pre>
        </div>
      ))}
      {evidence.length > shown.length && (
        <button className="link" onClick={() => setShowAll(true)}>
          Show all {evidence.length} retrieved chunks
        </button>
      )}
    </section>
  );
}
