import { useCallback, useEffect, useState } from "react";
import { createInvestigation, listInvestigations, listRepos } from "./api.js";
import InvestigationDetail from "./components/InvestigationDetail.jsx";
import Repos from "./components/Repos.jsx";
import "./styles.css";

function timeAgo(iso) {
  if (!iso) return "—";
  const secs = Math.max(0, (Date.now() - new Date(iso + "Z").getTime()) / 1000);
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

export default function App() {
  const [selected, setSelected] = useState(null);
  const [repos, setRepos] = useState([]);
  const [runs, setRuns] = useState([]);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [r, i] = await Promise.all([listRepos(), listInvestigations()]);
      setRepos(r);
      setRuns(i);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  async function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const inv = await createInvestigation(url.trim());
      setUrl("");
      await refresh();
      setSelected(inv.id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wrap">
      <header className="top">
        <h1>Bug Investigator</h1>
        <p>Paste a GitHub issue. Get a cited root cause, a suggested fix and tests.</p>
      </header>

      {selected ? (
        <InvestigationDetail id={selected} onBack={() => setSelected(null)} />
      ) : (
        <>
          <section className="panel">
            <h2>Investigate an issue</h2>
            <form className="row" onSubmit={submit}>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://github.com/owner/repo/issues/123"
                required
              />
              <button disabled={busy}>{busy ? "Starting…" : "Investigate"}</button>
            </form>
            {error && <p className="error">{error}</p>}
          </section>

          <Repos repos={repos} onChange={refresh} />

          <section className="panel">
            <h2>Runs</h2>
            {runs.length === 0 ? (
              <p className="empty">Nothing yet.</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Status</th>
                    <th>Passes</th>
                    <th>Confidence</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id} className="clickable" onClick={() => setSelected(r.id)}>
                      <td className="mono">{r.id}</td>
                      <td>
                        <span className={`badge ${r.status}`}>{r.status}</span>
                      </td>
                      <td>{r.iterations}</td>
                      <td>{r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "—"}</td>
                      <td className="muted">{timeAgo(r.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </>
      )}
    </div>
  );
}
