/**
 * MedFed AI — backend API client.
 * Single layer of fetch() calls with bearer token injection.
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type ApiError = { status: number; detail: string };

export class MedFedError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

function getToken(): string | null {
  return localStorage.getItem("medfed_token");
}

export function setToken(t: string | null) {
  if (t === null) {
    localStorage.removeItem("medfed_token");
  } else {
    localStorage.setItem("medfed_token", t);
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string> | undefined),
  };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new MedFedError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

async function requestFormData<T>(path: string, file: File, extra: Record<string, string> = {}): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  for (const [k, v] of Object.entries(extra)) {
    if (v !== undefined && v !== null) form.append(k, v);
  }
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new MedFedError(res.status, detail);
  }
  return (await res.json()) as T;
}

// ──────────────────────────────────────────────────────────────────────
// Auth
// ──────────────────────────────────────────────────────────────────────
export type Role = "doctor" | "researcher" | "institution_admin" | "platform_admin";
export type PublicUser = {
  username: string;
  full_name: string;
  hospital_id: string;
  role: Role;
  role_label: string;
};

export const auth = {
  async login(username: string, password: string) {
    return request<{ access_token: string; user: PublicUser }>(
      "/api/auth/login",
      {
        method: "POST",
        body: JSON.stringify({ username, password }),
      },
    );
  },
  me() {
    return request<PublicUser>("/api/auth/me");
  },
  roles() {
    return request<Record<string, { label: string; description: string; permissions: string[] }>>(
      "/api/auth/roles",
    );
  },
};

// ──────────────────────────────────────────────────────────────────────
// System
// ──────────────────────────────────────────────────────────────────────
export const system = {
  health() {
    return request<{ status: string; service: string; version: string; study_type: string; dataset: string }>(
      "/api/health",
    );
  },
  modelStatus() {
    return request<{
      available: boolean;
      device: string;
      loaded_for_seconds: number;
      error: string | null;
      registry: {
        current_version: string | null;
        path: string | null;
        round: number | null;
        status: string;
        metrics: { f1: number | null; roc_auc: number | null; loss: number | null };
        metadata: Record<string, unknown>;
        created_at: string | null;
        updated_at?: string;
      };
    }>("/api/model/status");
  },
};

// ──────────────────────────────────────────────────────────────────────
// Prediction
// ──────────────────────────────────────────────────────────────────────
export type Prediction = {
  top_predictions: { label: string; probability: number }[];
  all_predictions: { label: string; probability: number }[];
  model_version: string | null;
  model_round: number | null;
  disclaimer: string;
  study_type: string;
};

export type Explanation = {
  predicted_label: string;
  predicted_confidence: number;
  explained_class: string;
  explained_class_index: number;
  explained_class_confidence: number;
  gradcam_png_base64: string | null;
  caption: string;
  disclaimer: string;
};

export const predict = {
  predict(file: File) {
    return requestFormData<Prediction>("/api/predict", file);
  },
  explain(file: File, targetClass?: string) {
    return requestFormData<Explanation>("/api/explain", file, targetClass ? { target_class: targetClass } : {});
  },
};

// ──────────────────────────────────────────────────────────────────────
// Models
// ──────────────────────────────────────────────────────────────────────
export type ModelVersion = {
  version: string;
  round: number | null;
  metrics: { f1: number | null; roc_auc: number | null; loss: number | null };
  metadata: { strategy: string | null; privacy: string | null };
  created_at: string;
  path: string;
};
export type ModelsList = {
  current_version: string | null;
  versions: ModelVersion[];
};

export const models = {
  list() {
    return request<ModelsList>("/api/models");
  },
  get(version: string) {
    return request<{ version: string; path: string; round: number | null; metrics: Record<string, number | null> }>(
      `/api/models/${version}`,
    );
  },
  deploy(version: string, confirm = true) {
    return request<{ available: boolean; device: string; loaded_for_seconds: number; registry: Record<string, unknown> }>(
      `/api/models/${version}/deploy`,
      {
        method: "POST",
        body: JSON.stringify({ confirm }),
      },
    );
  },
};

// ──────────────────────────────────────────────────────────────────────
// Training
// ──────────────────────────────────────────────────────────────────────
export type TrainingJob = {
  id: string;
  status: "pending" | "running" | "completed" | "failed" | "stopped";
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  strategy: string;
  privacy: string;
  rounds: number;
  local_epochs: number;
  batch_size: number;
  lr: number;
  mu: number;
  max_batches: number;
  seed: number;
  hospital_nodes: string[];
  requested_by: string;
  current_round: number;
  progress_pct: number;
  per_round: { round: number | null; global_macro_f1: number | null; global_roc_auc: number | null; round_duration_sec: number | null; straggler_node: string | null; client_f1s: Record<string, number> }[];
  result: { summary_path?: string; best_global_macro_f1?: number; final_global_macro_f1?: number; final_global_roc_auc?: number; total_wall_clock_sec?: number } | null;
  error: string | null;
  log_tail: string[];
};

export const training = {
  start(req: {
    strategy: string;
    privacy: string;
    rounds: number;
    local_epochs: number;
    batch_size: number;
    lr: number;
    mu: number;
    max_batches: number;
    seed: number;
    hospital_nodes: string[];
    confirm: boolean;
  }) {
    return request<TrainingJob>("/api/training/start", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },
  stop(jobId: string) {
    return request<TrainingJob>(`/api/training/stop?job_id=${encodeURIComponent(jobId)}`, {
      method: "POST",
    });
  },
  list() {
    return request<{ runs: TrainingJob[] }>("/api/training/runs");
  },
  get(jobId: string) {
    return request<TrainingJob>(`/api/training/runs/${encodeURIComponent(jobId)}`);
  },
  status() {
    return request<{ jobs: TrainingJob[] }>("/api/training/status");
  },
};

// ──────────────────────────────────────────────────────────────────────
// Nodes
// ──────────────────────────────────────────────────────────────────────
export type NodeInfo = {
  node_id: string;
  hospital_id: string;
  total_samples: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  status: "connected" | "disconnected";
  data_locality: "isolated";
  data_locality_verified: boolean;
};

export const nodes = {
  list() {
    return request<{ nodes: NodeInfo[] }>("/api/nodes");
  },
  get(id: string) {
    return request<NodeInfo & Record<string, unknown>>(`/api/nodes/${id}`);
  },
};
