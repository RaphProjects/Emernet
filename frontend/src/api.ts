const API = import.meta.env.VITE_API_BASE_URL;
const CLIENT_ID_KEY = 'emernet-client-id';

export function getClientId() {
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = crypto.randomUUID ? crypto.randomUUID() : `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

function clientHeaders(extra: Record<string, string> = {}) {
  return { ...extra, 'X-Emernet-Client-Id': getClientId() };
}

export async function generateArch(moduleSet = 'Unified', archSize = 12) {
  const params = new URLSearchParams({
    module_set: moduleSet,
    arch_size: String(archSize),
  });
  const res = await fetch(`${API}/api/generate?${params.toString()}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function uploadArch(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API}/api/upload_arch`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export async function uploadGnn(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API}/api/upload_gnn`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function uploadLgbm(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API}/api/upload_lgbm`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function uploadPolicy(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API}/api/upload_policy`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function downloadArch(archId: string, filename: string) {
  const res = await fetch(`${API}/api/download_arch/${archId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.pkl') ? filename : `${filename}.pkl`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function downloadArchPython(archId: string, filename: string) {
  const res = await fetch(`${API}/api/download_arch_python/${archId}`);
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.py') ? filename : `${filename}.py`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function downloadGnn(gnnId: string, filename = 'rl_search_gnn') {
  const res = await fetch(`${API}/api/download_gnn/${gnnId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.pkl') ? filename : `${filename}.pkl`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function downloadLgbm(lgbmId: string, filename = 'rl_search_lgbm') {
  const res = await fetch(`${API}/api/download_lgbm/${lgbmId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.pkl') ? filename : `${filename}.pkl`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function downloadPolicy(policyId: string, filename = 'evolution_policy') {
  const res = await fetch(`${API}/api/download_policy/${policyId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.pkl') ? filename : `${filename}.pkl`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function getArenaNormalization() {
  const res = await fetch(`${API}/api/arena_normalization`, { headers: clientHeaders() });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function updateArenaNormalization(normalization: Record<string, any>) {
  const res = await fetch(`${API}/api/arena_normalization`, {
    method: 'POST',
    headers: clientHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ normalization }),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function getSpeedBalance() {
  const res = await fetch(`${API}/api/speed_balance`, { headers: clientHeaders() });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function updateSpeedBalance(speedBalance: Record<string, number>) {
  const res = await fetch(`${API}/api/speed_balance`, {
    method: 'POST',
    headers: clientHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ speed_balance: speedBalance }),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function getPolicyInfo(policyId: string) {
  const res = await fetch(`${API}/api/policy_info/${policyId}`);
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function predictArchWithPolicy(payload: {
  arch_id: string;
  policy_id: string;
  module_set?: string;
  speed_bal?: number;
  opp_simp_bal?: number;
}) {
  const res = await fetch(`${API}/api/policy_predict_arch`, {
    method: 'POST',
    headers: clientHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ ...payload, client_id: getClientId() }),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function getRealDatasets() {
  const res = await fetch(`${API}/api/real_datasets`);
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function runRealDatasetTest(payload: Record<string, any>) {
  const res = await fetch(`${API}/api/real_dataset_test`, {
    method: 'POST',
    headers: clientHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ ...payload, client_id: getClientId() }),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function runRealDatasetUploadTest(payload: Record<string, any>, filesByDataset: Record<string, File[]>) {
  const form = new FormData();
  form.append('arch_id', payload.arch_id);
  form.append('datasets', JSON.stringify(payload.datasets ?? []));
  form.append('paths', JSON.stringify(payload.paths ?? {}));
  form.append('target_columns', JSON.stringify(payload.target_columns ?? {}));
  form.append('max_iter', String(payload.max_iter ?? 80));
  form.append('subsample', String(payload.subsample ?? 2000));
  form.append('test_size', String(payload.test_size ?? 0.3));
  const manifest: Array<{ dataset_id: string; relative_path: string }> = [];
  Object.entries(filesByDataset).forEach(([datasetId, files]) => {
    files.forEach(file => {
      const rel = (file as any).webkitRelativePath || file.name;
      manifest.push({ dataset_id: datasetId, relative_path: rel });
      form.append('files', file, rel);
    });
  });
  form.append('upload_manifest', JSON.stringify(manifest));
  const res = await fetch(`${API}/api/real_dataset_upload_test`, {
    method: 'POST',
    headers: clientHeaders(),
    body: form,
  });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export async function mutateArch(archId: string, action: string, params: Record<string, any> = {}) {
  const res = await fetch(`${API}/api/mutate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arch_id: archId, action, params }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function crossoverArchs(archAId: string, archBId: string, splitNode?: number) {
  const res = await fetch(`${API}/api/crossover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arch_a_id: archAId, arch_b_id: archBId, split_node: splitNode }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getNodeParams(archId: string, nodeId: number) {
  const res = await fetch(`${API}/api/arch_params/${archId}/${nodeId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function importArchSubgraph(archId: string, file: File) {
  const formData = new FormData();
  formData.append('arch_id', archId);
  formData.append('file', file);
  const res = await fetch(`${API}/api/import_arch_subgraph`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export interface RlSearchConfig {
  mode?: 'legacy' | 'train_encoder' | 'evolve';
  n_phase_a_episodes: number;
  n_phase_b_episodes: number;
  n_candidates_per_step: number;
  tournament_size?: number;
  simp_opp_bal?: number;
  n_gnn_epochs?: number;
  acceptance_temperature?: number;
  train_on_tournament_archs?: boolean;
  module_set: string;
  retrain_frequency: number;
  arch_size: number;
  n_fights: number;
  dataset_size: number;
  gnn_id?: string | null;
  lgbm_id?: string | null;
  policy_id?: string | null;
  base_arch_id?: string | null;
  skip_phase_b?: boolean;
}

export function startRlSearch(
  config: RlSearchConfig,
  callbacks: {
    onStatus?: (msg: string) => void;
    onPhaseAEpisode?: (data: any) => void;
    onPhaseAComplete?: (data: any) => void;
    onPhaseBEpisode?: (data: any) => void;
    onPhaseBCandidates?: (data: any) => void;
    onPhaseBComplete?: (data: any) => void;
    onDiagnostic?: (data: any) => void;
    onArenaEvent?: (data: any) => void;
    onTournamentEvent?: (data: any) => void;
    onEncoderEpoch?: (data: any) => void;
    onEncoderDone?: (data: any) => void;
    onEvolveInitial?: (data: any) => void;
    onEvolveCandidates?: (data: any) => void;
    onEvolveTesting?: (data: any) => void;
    onEvolveEpoch?: (data: any) => void;
    onEvolveTopArchitectures?: (data: any) => void;
    onEvolveDone?: (data: any) => void;
    onDone?: (data: any) => void;
    onError?: (msg: string) => void;
  },
): WebSocket {
  const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE_URL}/ws/rl_search`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ ...config, client_id: getClientId() }));
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'status') {
      callbacks.onStatus?.(data.message);
    } else if (data.type === 'phase_a_episode') {
      callbacks.onPhaseAEpisode?.(data);
    } else if (data.type === 'phase_a_done') {
      callbacks.onPhaseAComplete?.(data);
    } else if (data.type === 'phase_b_episode') {
      callbacks.onPhaseBEpisode?.(data);
    } else if (data.type === 'phase_b_candidates') {
      callbacks.onPhaseBCandidates?.(data);
    } else if (data.type === 'phase_b_done') {
      callbacks.onPhaseBComplete?.(data);
    } else if (data.type === 'rl_diagnostic') {
      callbacks.onDiagnostic?.(data);
    } else if (data.type === 'arena_eval_start' || data.type === 'arena_fight_result' || data.type === 'arena_eval_done') {
      callbacks.onArenaEvent?.(data);
    } else if (data.type === 'rl_tournament_init' || data.type === 'rl_tournament_fight' || data.type === 'rl_tournament_done') {
      callbacks.onTournamentEvent?.(data);
    } else if (data.type === 'encoder_epoch') {
      callbacks.onEncoderEpoch?.(data);
    } else if (data.type === 'encoder_done') {
      callbacks.onEncoderDone?.(data);
    } else if (data.type === 'evolve_initial') {
      callbacks.onEvolveInitial?.(data);
    } else if (data.type === 'evolve_candidates') {
      callbacks.onEvolveCandidates?.(data);
    } else if (data.type === 'evolve_testing') {
      callbacks.onEvolveTesting?.(data);
    } else if (data.type === 'evolve_epoch') {
      callbacks.onEvolveEpoch?.(data);
    } else if (data.type === 'evolve_top_architectures') {
      callbacks.onEvolveTopArchitectures?.(data);
    } else if (data.type === 'evolve_done') {
      callbacks.onEvolveDone?.(data);
    } else if (data.type === 'done') {
      callbacks.onDone?.(data);
    } else if (data.type === 'error') {
      callbacks.onError?.(data.message ?? 'RL search failed.');
    }
  };

  ws.onerror = () => {
    callbacks.onError?.('WebSocket connection error — is the backend running?');
  };

  ws.onclose = () => {
    callbacks.onStatus?.('Connection closed.');
  };

  return ws;
}
