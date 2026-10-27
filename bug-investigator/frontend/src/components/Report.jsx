import { reportUrl } from "../api.js";

export default function Report({ investigation }) {
  const r = investigation.report;
  if (!r) return null;

  return (
    <section className="panel">
      <h2>Report</h2>

      {r.low_confidence && (
        <p className="notice">
          The critic never accepted this analysis. Treat it as a lead, not an answer.
        </p>
      )}

      <dl className="kv">
        <div>
          <dt>Confidence</dt>
          <dd>{Math.round((r.confidence ?? 0) * 100)}%</dd>
        </div>
        <div>
          <dt>Critic passes</dt>
          <dd>{investigation.iterations}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd className="mono">{investigation.llm_model || "—"}</dd>
        </div>
      </dl>

      <h3 style={{ fontSize: 15, margin: "0 0 6px" }}>Root cause</h3>
      <p style={{ marginTop: 0 }}>{r.root_cause}</p>

      {r.affected_files?.length > 0 && (
        <>
          <h3 style={{ fontSize: 15, margin: "16px 0 6px" }}>Affected files</h3>
          <ul className="mono" style={{ margin: 0, paddingLeft: 20 }}>
            {r.affected_files.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </>
      )}

      {r.fixes?.length > 0 && (
        <>
          <h3 style={{ fontSize: 15, margin: "16px 0 6px" }}>Suggested fix</h3>
          {r.fixes.map((f, i) => (
            <div key={i}>
              <p className="mono muted" style={{ margin: 0 }}>{f.file_path}</p>
              <pre className="block">{f.diff}</pre>
              {f.explanation && <p style={{ fontSize: 14 }}>{f.explanation}</p>}
            </div>
          ))}
        </>
      )}

      {r.tests?.length > 0 && (
        <>
          <h3 style={{ fontSize: 15, margin: "16px 0 6px" }}>Suggested tests</h3>
          {r.tests.map((t, i) => (
            <div key={i}>
              <p className="mono muted" style={{ margin: 0 }}>{t.name}</p>
              <pre className="block">{t.code}</pre>
              {t.rationale && <p style={{ fontSize: 14 }}>{t.rationale}</p>}
            </div>
          ))}
        </>
      )}

      <p style={{ marginBottom: 0 }}>
        <a href={reportUrl(investigation.id)} download={`investigation-${investigation.id}.md`}>
          Download Markdown
        </a>
      </p>
    </section>
  );
}
