const SUMMARY = {
  planner: (o) => `${(o.queries || []).length} queries · ${(o.hypotheses || []).length} hypotheses`,
  investigator: (o) =>
    `${o.evidence_retrieved ?? 0} chunks · confidence ${Math.round((o.confidence ?? 0) * 100)}%`,
  fix: (o) => `${(o.tests || []).length} test(s) suggested`,
  critic: (o) => o.verdict || "",
};

export default function Timeline({ steps }) {
  if (!steps.length) return <p className="empty">No steps recorded yet.</p>;
  return (
    <ol className="timeline">
      {steps.map((s) => {
        const verdict = s.agent === "critic" ? s.output?.verdict : null;
        return (
          <li key={s.seq} className={verdict || ""}>
            <div className="step-head">
              <strong>{s.agent}</strong>
              <span className="muted">pass {s.iteration + 1}</span>
              <span className="muted">{SUMMARY[s.agent]?.(s.output || {}) || ""}</span>
              {s.latency_ms != null && <span className="muted">{s.latency_ms} ms</span>}
            </div>
            {s.agent === "critic" && s.output?.feedback && (
              <p className="muted" style={{ margin: "4px 0 0", fontSize: 13 }}>
                “{s.output.feedback}”
              </p>
            )}
            <details>
              <summary>raw output</summary>
              <pre>{JSON.stringify(s.output, null, 2)}</pre>
            </details>
          </li>
        );
      })}
    </ol>
  );
}
