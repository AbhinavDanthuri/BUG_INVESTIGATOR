import { useEffect, useState } from "react";
import { getInvestigation } from "../api.js";
import EvidenceList from "./EvidenceList.jsx";
import Report from "./Report.jsx";
import Timeline from "./Timeline.jsx";

const ACTIVE = new Set(["queued", "running"]);

export default function InvestigationDetail({ id, onBack }) {
  const [inv, setInv] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    let timer;

    async function poll() {
      try {
        const data = await getInvestigation(id);
        if (!alive) return;
        setInv(data);
        setError("");
        if (ACTIVE.has(data.status)) timer = setTimeout(poll, 2000);
      } catch (err) {
        if (alive) setError(err.message);
      }
    }
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id]);

  if (error) return <p className="error">{error}</p>;
  if (!inv) return <p className="empty">Loading…</p>;

  return (
    <>
      <button className="ghost" onClick={onBack} style={{ marginTop: 20 }}>
        ← All runs
      </button>

      <section className="panel">
        <div className="step-head">
          <span className={`badge ${inv.status}`}>{inv.status}</span>
          <strong className="mono">
            {inv.repo_full_name}#{inv.issue?.number}
          </strong>
        </div>
        <h2 style={{ margin: "10px 0 4px", textTransform: "none", fontSize: 17, color: "var(--text)" }}>
          {inv.issue?.title}
        </h2>
        {inv.issue?.html_url && (
          <a className="mono" href={inv.issue.html_url} target="_blank" rel="noreferrer">
            view on GitHub
          </a>
        )}
        {ACTIVE.has(inv.status) && (
          <p className="muted" style={{ marginBottom: 0 }}>
            Investigating… this page updates itself.
          </p>
        )}
        {inv.error && <p className="error">{inv.error}</p>}
      </section>

      <Report investigation={inv} />

      <section className="panel">
        <h2>Agent timeline</h2>
        <Timeline steps={inv.steps} />
      </section>

      <EvidenceList evidence={inv.evidence} />
    </>
  );
}
