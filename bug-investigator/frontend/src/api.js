async function request(path, options) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const listRepos = () => request("/api/repos");
export const registerRepo = (repo_url) =>
  request("/api/repos", { method: "POST", body: JSON.stringify({ repo_url }) });
export const reindexRepo = (id) => request(`/api/repos/${id}/reindex`, { method: "POST" });

export const createInvestigation = (issue_url) =>
  request("/api/investigations", { method: "POST", body: JSON.stringify({ issue_url }) });
export const listInvestigations = () => request("/api/investigations");
export const getInvestigation = (id) => request(`/api/investigations/${id}`);
export const reportUrl = (id) => `/api/investigations/${id}/report.md`;
