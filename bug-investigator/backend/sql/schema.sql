-- Generated from app/db/models.py by scripts/export_schema.py. Do not edit by hand.

CREATE TABLE repositories (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	owner VARCHAR(100) NOT NULL, 
	name VARCHAR(100) NOT NULL, 
	default_branch VARCHAR(100) NOT NULL, 
	clone_url VARCHAR(300) NOT NULL, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	CONSTRAINT uq_repo_owner_name UNIQUE (owner, name)
);

CREATE TABLE eval_cases (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	repo_id BIGINT NOT NULL, 
	issue_number INTEGER NOT NULL, 
	gold_pr_number INTEGER, 
	gold_files JSON NOT NULL, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	CONSTRAINT uq_eval_repo_issue UNIQUE (repo_id, issue_number), 
	FOREIGN KEY(repo_id) REFERENCES repositories (id) ON DELETE CASCADE
);

CREATE TABLE index_versions (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	repo_id BIGINT NOT NULL, 
	commit_sha VARCHAR(40) NOT NULL, 
	embedding_model VARCHAR(200) NOT NULL, 
	embedding_dim INTEGER NOT NULL, 
	chunk_count INTEGER NOT NULL, 
	faiss_path VARCHAR(500), 
	status ENUM('building','ready','failed') NOT NULL, 
	error TEXT, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	finished_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(repo_id) REFERENCES repositories (id) ON DELETE CASCADE
);

CREATE INDEX ix_indexver_repo_sha ON index_versions (repo_id, commit_sha);

CREATE TABLE issues (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	repo_id BIGINT NOT NULL, 
	number INTEGER NOT NULL, 
	title VARCHAR(500) NOT NULL, 
	body LONGTEXT, 
	state VARCHAR(20) NOT NULL, 
	labels JSON, 
	author VARCHAR(100), 
	html_url VARCHAR(300) NOT NULL, 
	created_at_gh DATETIME, 
	closed_at_gh DATETIME, 
	fetched_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	CONSTRAINT uq_issue_repo_number UNIQUE (repo_id, number), 
	FOREIGN KEY(repo_id) REFERENCES repositories (id) ON DELETE CASCADE
);

CREATE TABLE chunks (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	repo_id BIGINT NOT NULL, 
	index_version_id BIGINT NOT NULL, 
	source_type ENUM('code','doc','issue','pr','commit') NOT NULL, 
	source_ref VARCHAR(512) NOT NULL, 
	start_line INTEGER, 
	end_line INTEGER, 
	text LONGTEXT NOT NULL, 
	content_hash VARCHAR(64) NOT NULL, 
	meta JSON, 
	PRIMARY KEY (id), 
	FOREIGN KEY(repo_id) REFERENCES repositories (id) ON DELETE CASCADE, 
	FOREIGN KEY(index_version_id) REFERENCES index_versions (id) ON DELETE CASCADE
);

CREATE INDEX ix_chunk_repo_ref ON chunks (repo_id, source_ref);

CREATE INDEX ix_chunk_version_type ON chunks (index_version_id, source_type);

CREATE TABLE investigations (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	repo_id BIGINT NOT NULL, 
	issue_id BIGINT NOT NULL, 
	index_version_id BIGINT, 
	status ENUM('queued','running','completed','failed') NOT NULL, 
	iterations INTEGER NOT NULL, 
	max_iterations INTEGER NOT NULL, 
	llm_provider VARCHAR(50), 
	llm_model VARCHAR(100), 
	prompt_version VARCHAR(20), 
	tokens_in INTEGER NOT NULL, 
	tokens_out INTEGER NOT NULL, 
	confidence FLOAT, 
	error TEXT, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	started_at DATETIME, 
	finished_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(repo_id) REFERENCES repositories (id), 
	FOREIGN KEY(issue_id) REFERENCES issues (id), 
	FOREIGN KEY(index_version_id) REFERENCES index_versions (id)
);

CREATE INDEX ix_inv_status_created ON investigations (status, created_at);

CREATE TABLE agent_steps (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	investigation_id BIGINT NOT NULL, 
	iteration INTEGER NOT NULL, 
	seq INTEGER NOT NULL, 
	agent ENUM('planner','investigator','fix','critic') NOT NULL, 
	input JSON, 
	output JSON, 
	tokens_in INTEGER NOT NULL, 
	tokens_out INTEGER NOT NULL, 
	latency_ms INTEGER, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	FOREIGN KEY(investigation_id) REFERENCES investigations (id) ON DELETE CASCADE
);

CREATE INDEX ix_step_inv_seq ON agent_steps (investigation_id, seq);

CREATE TABLE critic_reviews (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	investigation_id BIGINT NOT NULL, 
	iteration INTEGER NOT NULL, 
	verdict ENUM('accept','revise') NOT NULL, 
	scores JSON, 
	feedback TEXT, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	FOREIGN KEY(investigation_id) REFERENCES investigations (id) ON DELETE CASCADE
);

CREATE INDEX ix_review_inv ON critic_reviews (investigation_id, iteration);

CREATE TABLE eval_results (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	eval_case_id BIGINT NOT NULL, 
	investigation_id BIGINT, 
	`system` VARCHAR(30) NOT NULL, 
	hit_at_1 BOOL NOT NULL, 
	hit_at_3 BOOL NOT NULL, 
	file_recall FLOAT NOT NULL, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	FOREIGN KEY(eval_case_id) REFERENCES eval_cases (id) ON DELETE CASCADE, 
	FOREIGN KEY(investigation_id) REFERENCES investigations (id) ON DELETE SET NULL
);

CREATE TABLE reports (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	investigation_id BIGINT NOT NULL, 
	root_cause TEXT NOT NULL, 
	summary TEXT, 
	confidence FLOAT NOT NULL, 
	affected_files JSON, 
	low_confidence BOOL NOT NULL, 
	report_md LONGTEXT NOT NULL, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	UNIQUE (investigation_id), 
	FOREIGN KEY(investigation_id) REFERENCES investigations (id) ON DELETE CASCADE
);

CREATE TABLE evidence (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	investigation_id BIGINT NOT NULL, 
	step_id BIGINT, 
	chunk_id BIGINT NOT NULL, 
	`rank` INTEGER NOT NULL, 
	score FLOAT NOT NULL, 
	cited_in_report BOOL NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(investigation_id) REFERENCES investigations (id) ON DELETE CASCADE, 
	FOREIGN KEY(step_id) REFERENCES agent_steps (id) ON DELETE SET NULL, 
	FOREIGN KEY(chunk_id) REFERENCES chunks (id)
);

CREATE INDEX ix_evidence_inv ON evidence (investigation_id);

CREATE TABLE fix_suggestions (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	report_id BIGINT NOT NULL, 
	file_path VARCHAR(512) NOT NULL, 
	diff LONGTEXT NOT NULL, 
	explanation TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(report_id) REFERENCES reports (id) ON DELETE CASCADE
);

CREATE TABLE test_suggestions (
	id BIGINT NOT NULL AUTO_INCREMENT, 
	report_id BIGINT NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	code LONGTEXT NOT NULL, 
	rationale TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(report_id) REFERENCES reports (id) ON DELETE CASCADE
);
