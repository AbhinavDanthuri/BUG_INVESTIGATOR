import { useState } from "react";
import { registerRepo, reindexRepo } from "../api.js";

function indexLabel(ix) {
  if (!ix) return <span className="muted">not indexed</span>;
  if (ix.status === "building") return <span className="badge running">building</span>;
  if (ix.status === "failed")
    return <span className="badge failed" title={ix.error || ""}>failed</span>;
  return (
    <span>
      <span className="badge completed">ready</span>{" "}
      <span className="muted mono">
        {ix.chunk_count.toLocaleString()} chunks @ {ix.commit_sha.slice(0, 7)}
      </span>
    </span>
  );
}

export default function Repos({ repos, onChange }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await registerRepo(url.trim());
      setUrl("");
      onChange();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function reindex(id) {
    try {
      await reindexRepo(id);
      onChange();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section className="panel">
      <h2>Repositories</h2>
      <form className="row" onSubmit={submit}>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="owner/repo or https://github.com/owner/repo"
          required
        />
        <button disabled={busy}>{busy ? "Indexing…" : "Add & index"}</button>
      </form>
      {error && <p className="error">{error}</p>}

      {repos.length === 0 ? (
        <p className="empty" style={{ marginTop: 14 }}>
          No repositories yet. Add one to build its search index.
        </p>
      ) : (
        <table style={{ marginTop: 14 }}>
          <thead>
            <tr>
              <th>Repository</th>
              <th>Index</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {repos.map((r) => (
              <tr key={r.id}>
                <td className="mono">
                  {r.owner}/{r.name}
                </td>
                <td>{indexLabel(r.latest_index)}</td>
                <td style={{ textAlign: "right" }}>
                  <button className="ghost" onClick={() => reindex(r.id)}>
                    Re-index
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
