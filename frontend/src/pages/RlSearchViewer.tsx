import { useCallback, useEffect, useRef, useState } from 'react';
import type React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  ReactFlow, Background, Controls, ReactFlowProvider, applyNodeChanges,
} from '@xyflow/react';
import type { Edge, Node, OnNodesChange } from '@xyflow/react';
import {
  startRlSearch, downloadArch, downloadGnn, downloadPolicy,
  uploadArch, uploadGnn, uploadPolicy,
} from '../api';
import { INPUT_STYLE, MODULE_COLORS, MODULE_SET_OPTIONS } from '../theme';
import LooseNumberInput from '../components/LooseNumberInput';

type Mode = 'train_encoder' | 'evolve';
type RunStatus = 'idle' | 'running' | 'done' | 'error';

interface TournamentArch {
  id: number;
  name: string;
  n_nodes: number;
  n_params: number;
  score?: number;
  learnability?: number;
  speed?: number;
  fight_count?: number;
  graph?: any;
}

interface FightLog {
  fight: number;
  total: number;
  i: number;
  j: number;
  score_i: number;
  score_j: number;
  failed?: boolean;
  phase?: string;
  epoch?: number;
  eval_role?: string;
}

interface EncoderPoint {
  epoch: number;
  avg_delta?: number | null;
  gnn_loss?: number;
  best_score?: number;
}

interface EvolvePoint {
  epoch: number;
  predicted_score?: number | null;
  true_score: number;
  prediction_error?: number | null;
  best_score: number;
  mutation_type: string;
}

interface CandidateRow {
  rank: number;
  mutation_type: string;
  target_module_type: string;
  mutation_sequence?: string[];
  n_mutation_steps?: number;
  predicted_components?: {
    learnability?: number;
    speed?: number;
    opp_simp_raw?: number;
    opp_simp_bonus?: number;
  } | null;
  predicted_score?: number | null;
  n_nodes: number;
  n_params: number;
}

interface VersionRow {
  version: number | null;
  epoch?: number;
  accepted?: boolean;
  exploratory?: boolean;
  mutation_type: string;
  target_module_type: string;
  mutation_sequence?: string[];
  n_mutation_steps?: number;
  arena_score: number;
  previous_arena_score: number;
  arena_delta: number;
  gnn_score?: number | null;
  lgbm_predicted_score?: number | null;
  prediction_error?: number | null;
  acceptance_probability?: number | null;
  true_learnability?: number;
  true_speed?: number;
  true_opp_simp_bonus?: number;
  true_opp_simp_raw_bonus?: number;
  predicted_components?: {
    learnability?: number;
    speed?: number;
    opp_simp_raw?: number;
    opp_simp_bonus?: number;
  } | null;
  n_nodes: number;
  n_params: number;
}

interface StoredVersion {
  version: number;
  label: string;
  score: number;
  arena_delta?: number | null;
  accepted?: boolean;
  exploratory?: boolean;
  acceptance_probability?: number | null;
  true_learnability?: number | null;
  true_speed?: number | null;
  true_opp_simp_bonus?: number | null;
  true_opp_simp_raw_bonus?: number | null;
  prediction_error?: number | null;
  arch_id: string;
  graph: any;
}

interface TopArchitecture {
  rank: number;
  role: 'evolved' | 'met';
  source: string;
  epoch: number;
  score: number;
  learnability?: number | null;
  speed?: number | null;
  opp_simp_raw?: number | null;
  n_nodes: number;
  n_params: number;
  arch_id: string;
  graph?: any;
}

const panel: React.CSSProperties = {
  background: 'var(--panel-bg)',
  border: '1px solid var(--glass-border)',
  borderRadius: '8px',
  boxShadow: 'var(--window-shadow)',
  backdropFilter: 'blur(14px)',
};

const input: React.CSSProperties = {
  ...INPUT_STYLE,
  minHeight: 38,
};

function fmt(v?: number | null, digits = 4) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '-';
}

function nodeStyle(type: string): React.CSSProperties {
  const color = MODULE_COLORS[type] ?? '#e2e8f0';
  return {
    background: `${color}e6`,
    border: '1px solid rgba(0,0,0,0.55)',
    borderRadius: '8px',
    padding: '5px 12px',
    fontSize: 11,
    fontWeight: 800,
    color: '#100800',
    whiteSpace: 'nowrap',
    fontFamily: 'inherit',
  };
}

function graphData(data: any): { nodes: Node[]; edges: Edge[] } {
  if (!data?.nodes) return { nodes: [], edges: [] };
  return {
    nodes: data.nodes.map((n: any) => ({
      id: String(n.id),
      position: { x: n.x ?? 0, y: n.y ?? 0 },
      data: { label: n.type },
      style: nodeStyle(n.type),
    })),
    edges: (data.edges ?? []).map((e: any, i: number) => ({
      id: `e${i}`,
      source: String(e.source),
      target: String(e.target),
      style: { stroke: 'rgba(var(--theme-primary-rgb),0.55)', strokeWidth: 1.5 },
      markerEnd: { type: 'arrowclosed' as const, color: 'rgba(255,122,24,0.65)' },
    })),
  };
}

function UploadButton({
  label, accept, onFile,
}: {
  label: string;
  accept: string;
  onFile: (file: File) => void;
}) {
  return (
    <label className="btn" style={{ cursor: 'pointer', textAlign: 'center' }}>
      {label}
      <input
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={e => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.currentTarget.value = '';
        }}
      />
    </label>
  );
}

function RlSearchViewerInner({ initialMode = 'train_encoder' }: { initialMode?: Mode }) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [status, setStatus] = useState<RunStatus>('idle');
  const [log, setLog] = useState('Choose a workflow.');
  const [error, setError] = useState<string | null>(null);

  const [epochsA, setEpochsA] = useState(5);
  const [epochsB, setEpochsB] = useState(10);
  const [tournamentSize, setTournamentSize] = useState(6);
  const [candidates, setCandidates] = useState(8);
  const [archSize, setArchSize] = useState(12);
  const [simpOppBal, setSimpOppBal] = useState(0.2);
  const [datasetSize, setDatasetSize] = useState(512);
  const [moduleSet, setModuleSet] = useState('Unified');
  const [retrainFreq, setRetrainFreq] = useState(1);
  const [gnnEpochs, setGnnEpochs] = useState(60);
  const [acceptanceTemperature, setAcceptanceTemperature] = useState(0.05);
  const [trainOnTournamentArchs, setTrainOnTournamentArchs] = useState(true);

  const [loadedGnn, setLoadedGnn] = useState<{ id: string; name: string } | null>(null);
  const [loadedPolicy, setLoadedPolicy] = useState<{ id: string; name: string; records: number } | null>(null);
  const [baseArch, setBaseArch] = useState<{ id: string; name: string } | null>(null);

  const [tournamentArchs, setTournamentArchs] = useState<TournamentArch[]>([]);
  const [fightLog, setFightLog] = useState<FightLog[]>([]);
  const [fightProgress, setFightProgress] = useState({ current: 0, total: 0 });
  const [encoderHistory, setEncoderHistory] = useState<EncoderPoint[]>([]);
  const [evolveHistory, setEvolveHistory] = useState<EvolvePoint[]>([]);
  const [candidateRows, setCandidateRows] = useState<CandidateRow[]>([]);
  const [versionRows, setVersionRows] = useState<VersionRow[]>([]);
  const [storedVersions, setStoredVersions] = useState<StoredVersion[]>([]);
  const [topArchitectures, setTopArchitectures] = useState<TopArchitecture[]>([]);
  const [graph, setGraph] = useState<{ nodes: Node[]; edges: Edge[] }>({ nodes: [], edges: [] });
  const [testedGraph, setTestedGraph] = useState<{ nodes: Node[]; edges: Edge[] }>({ nodes: [], edges: [] });
  const [epochProgress, setEpochProgress] = useState({ current: 0, total: 0 });
  const [initialScore, setInitialScore] = useState<number | null>(null);
  const [bestScore, setBestScore] = useState<number | null>(null);
  const [finalArchId, setFinalArchId] = useState<string | null>(null);
  const [finalGnnId, setFinalGnnId] = useState<string | null>(null);
  const [finalPolicyId, setFinalPolicyId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const pushLog = useCallback((message: string) => setLog(message), []);
  const onNodesChange: OnNodesChange = useCallback(changes => {
    setGraph(prev => ({ ...prev, nodes: applyNodeChanges(changes, prev.nodes) }));
  }, []);
  const onTestedNodesChange: OnNodesChange = useCallback(changes => {
    setTestedGraph(prev => ({ ...prev, nodes: applyNodeChanges(changes, prev.nodes) }));
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  const resetRun = () => {
    wsRef.current?.close();
    setStatus('idle');
    setError(null);
    setTournamentArchs([]);
    setFightLog([]);
    setFightProgress({ current: 0, total: 0 });
    setEncoderHistory([]);
    setEvolveHistory([]);
    setCandidateRows([]);
    setVersionRows([]);
    setStoredVersions([]);
    setTopArchitectures([]);
    setTestedGraph({ nodes: [], edges: [] });
    setEpochProgress({ current: 0, total: 0 });
    setInitialScore(null);
    setBestScore(null);
    setFinalArchId(null);
    setFinalGnnId(null);
    setFinalPolicyId(null);
    setLog('Ready.');
  };

  const start = () => {
    if (mode === 'evolve' && !loadedGnn && !loadedPolicy) {
      setStatus('error');
      setError('Upload a trained GNN or Evolution Policy before evolving an architecture.');
      return;
    }
    resetRun();
    setStatus('running');
    const ws = startRlSearch({
      mode,
      n_phase_a_episodes: epochsA,
      n_phase_b_episodes: epochsB,
      n_candidates_per_step: candidates,
      tournament_size: tournamentSize,
      simp_opp_bal: simpOppBal,
      n_gnn_epochs: gnnEpochs,
      acceptance_temperature: acceptanceTemperature,
      train_on_tournament_archs: trainOnTournamentArchs,
      module_set: moduleSet,
      retrain_frequency: retrainFreq,
      arch_size: archSize,
      n_fights: 1,
      dataset_size: datasetSize,
      gnn_id: loadedGnn?.id ?? null,
      policy_id: loadedPolicy?.id ?? null,
      base_arch_id: baseArch?.id ?? null,
    }, {
      onStatus: pushLog,
      onTournamentEvent: data => {
        if (data.type === 'rl_tournament_init') {
          setTournamentArchs(data.architectures ?? []);
          setFightProgress({ current: 0, total: data.total_fights ?? 0 });
          setFightLog([]);
          pushLog(`Tournament started: ${data.n_archs} architectures, ${data.total_fights} fights.`);
        } else if (data.type === 'rl_tournament_fight') {
          setFightProgress({ current: data.fight, total: data.total });
          setFightLog(prev => [...prev, data].slice(-120));
          setTournamentArchs(prev => prev.map((arch, idx) => ({
            ...arch,
            score: data.scores?.[idx],
            learnability: data.learnabilities?.[idx],
            speed: data.speeds?.[idx],
            fight_count: data.fight_counts?.[idx],
          })));
        } else if (data.type === 'rl_tournament_done') {
          setTournamentArchs(prev => prev.map((arch, idx) => ({
            ...arch,
            score: data.final_scores?.[idx],
            learnability: data.learnabilities?.[idx],
            speed: data.speeds?.[idx],
            fight_count: data.fight_counts?.[idx],
          })));
          pushLog('Tournament complete; training step is using the final scores.');
        }
      },
      onEncoderEpoch: data => {
        setEncoderHistory(prev => [...prev, {
          epoch: data.epoch,
          avg_delta: data.avg_delta,
          gnn_loss: data.gnn_loss,
          best_score: data.best_score,
        }]);
        setBestScore(data.best_score);
        if (data.best_arch_json) setGraph(graphData(data.best_arch_json));
        pushLog(`Encoder epoch ${data.epoch}: average prediction error ${fmt(data.avg_delta)}.`);
      },
      onEvolveInitial: data => {
        if (data.current_arch_json) setGraph(graphData(data.current_arch_json));
        if (data.pending_tournament || data.score == null) {
          pushLog('Initial architecture generated; starting its first tournament.');
        } else {
          setInitialScore(data.score);
          setBestScore(data.score);
          pushLog(`Initial architecture score: ${fmt(data.score)}.`);
        }
      },
      onEvolveCandidates: data => {
        setEpochProgress({ current: data.epoch ?? 0, total: data.n_total ?? epochsB });
        const rows = [...(data.candidates ?? [])]
          .sort((a, b) => (b.predicted_score ?? -Infinity) - (a.predicted_score ?? -Infinity));
        setCandidateRows(rows);
        pushLog(`Epoch ${data.epoch}: ${rows.length} valid mutations searched by ${data.used_model ? 'LGBM' : 'GNN fallback'}.`);
      },
      onEvolveTesting: data => {
        setEpochProgress({ current: data.epoch ?? 0, total: data.n_total ?? epochsB });
        if (data.tested_arch_json) setTestedGraph(graphData(data.tested_arch_json));
        const label = data.mutation_sequence?.length ? data.mutation_sequence.join(' -> ') : data.mutation_type;
        pushLog(`Epoch ${data.epoch}: testing ${label}, predicted ${fmt(data.predicted_score)}.`);
      },
      onEvolveEpoch: data => {
        setEvolveHistory(prev => [...prev, {
          epoch: data.epoch,
          predicted_score: data.predicted_score,
          true_score: data.true_score,
          prediction_error: data.prediction_error,
          best_score: data.best_score,
          mutation_type: data.mutation_type,
        }]);
        if (data.version_score) setVersionRows(prev => [...prev, data.version_score]);
        if (data.current_arch_json) setGraph(graphData(data.current_arch_json));
        setBestScore(data.best_score);
        const outcome = data.skipped ? 'skipped' : data.accepted ? 'accepted' : data.exploratory ? 'explored and rejected' : 'rejected';
        const added = data.tournament_records_added ? `, +${data.tournament_records_added} tournament records` : '';
        pushLog(`Epoch ${data.epoch}: ${data.mutation_type}, arena score ${fmt(data.true_score)} (${outcome}, p=${fmt(data.acceptance_probability, 2)}${added}).`);
      },
      onEvolveTopArchitectures: data => {
        setTopArchitectures(data.top_architectures ?? []);
      },
      onDone: data => {
        setStatus('done');
        setFinalArchId(data.arch_id ?? null);
        setFinalGnnId(data.gnn_id ?? null);
        setFinalPolicyId(data.policy_id ?? null);
        setStoredVersions(data.versions ?? []);
        setTopArchitectures(data.top_architectures ?? []);
        setBestScore(data.best_reward ?? bestScore);
        if (data.best_arch_json) setGraph(graphData(data.best_arch_json));
        pushLog(data.interrupted ? 'Workflow interrupted. Partial models and architectures are available.' : 'Workflow complete.');
      },
      onError: msg => {
        setStatus('error');
        setError(msg);
        pushLog(msg);
      },
    });
    wsRef.current = ws;
  };

  const sortedArchs = [...tournamentArchs].sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));
  const activeHistory = mode === 'train_encoder' ? encoderHistory : evolveHistory;
  const progressPct = fightProgress.total > 0 ? Math.round((fightProgress.current / fightProgress.total) * 100) : 0;
  const stopRun = () => {
    wsRef.current?.send(JSON.stringify({ type: 'stop' }));
    pushLog('Stop requested. Waiting for the current arena step to finish...');
  };

  return (
    <div style={{ padding: '24px', maxWidth: 1500, margin: '0 auto', maxHeight: 'calc(100vh - 72px)', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'center', marginBottom: 18 }}>
        <div>
          <h1 style={{ margin: 0, color: 'var(--text-primary)', fontSize: 28 }}>Architecture Training Arena</h1>
          <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>{log}</div>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {finalArchId && <button className="btn" onClick={() => downloadArch(finalArchId, 'evolved_architecture')}>Download Arch</button>}
          {finalGnnId && <button className="btn" onClick={() => downloadGnn(finalGnnId, 'arch_encoder_gnn')}>Download GNN</button>}
          {finalPolicyId && <button className="btn" onClick={() => downloadPolicy(finalPolicyId, 'evolution_policy')}>Download Policy</button>}
          {status === 'running' && <button className="btn btn-back" onClick={stopRun}>Stop</button>}
          {status !== 'running' && <button className="btn btn-back" onClick={resetRun}>Reset</button>}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: 14 }}>
        <aside style={{ ...panel, padding: 16, height: 'fit-content' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 16 }}>
            <button className={mode === 'train_encoder' ? 'btn' : 'btn btn-back'} onClick={() => setMode('train_encoder')}>Train Arch Encoder</button>
            <button className={mode === 'evolve' ? 'btn' : 'btn btn-back'} onClick={() => setMode('evolve')}>Evolve Architecture</button>
          </div>

          <div style={{ display: 'grid', gap: 12 }}>
            <Field label={mode === 'train_encoder' ? 'Tournament Epochs' : 'Evolution Epochs'}>
              <LooseNumberInput
                style={input}
                min={1}
                value={mode === 'train_encoder' ? epochsA : epochsB}
                onChange={value => mode === 'train_encoder' ? setEpochsA(value) : setEpochsB(value)}
                fallback={1}
              />
            </Field>
            <Field label="Tournament Size">
              <LooseNumberInput style={input} min={2} max={64} value={tournamentSize} onChange={setTournamentSize} fallback={2} />
            </Field>
            <Field label="Arch Size">
              <LooseNumberInput style={input} min={3} max={64} value={archSize} onChange={setArchSize} fallback={3} />
            </Field>
            <Field label="Opp Simp Bal">
              <LooseNumberInput style={input} step="0.05" value={simpOppBal} onChange={setSimpOppBal} />
            </Field>
            <Field label="Dataset Size">
              <LooseNumberInput style={input} min={64} value={datasetSize} onChange={setDatasetSize} fallback={64} />
            </Field>
            <Field label="Module Set">
              <select style={input} value={moduleSet} onChange={e => setModuleSet(e.target.value)}>
                {MODULE_SET_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </Field>
            {mode === 'train_encoder' ? (
              <Field label="GNN Training Epochs">
                <LooseNumberInput style={input} min={1} value={gnnEpochs} onChange={setGnnEpochs} fallback={1} />
              </Field>
            ) : (
              <>
                <Field label="Mutations per Epoch">
                  <LooseNumberInput style={input} min={1} value={candidates} onChange={setCandidates} fallback={1} />
                </Field>
                <Field label="LGBM Retrain Every">
                  <LooseNumberInput style={input} min={1} value={retrainFreq} onChange={setRetrainFreq} fallback={1} />
                </Field>
                <Field label="Acceptance Temperature">
                  <LooseNumberInput style={input} min={0} step="0.01" value={acceptanceTemperature} onChange={setAcceptanceTemperature} fallback={0} />
                </Field>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.35 }}>
                  <input
                    type="checkbox"
                    checked={trainOnTournamentArchs}
                    onChange={e => setTrainOnTournamentArchs(e.target.checked)}
                  />
                  Train policy on tournament architectures
                </label>
              </>
            )}
          </div>

          <div style={{ height: 1, background: 'var(--glass-border)', margin: '16px 0' }} />

          <div style={{ display: 'grid', gap: 8 }}>
            {mode === 'evolve' && (
              <>
                <UploadButton label={loadedGnn ? `GNN: ${loadedGnn.name}` : 'Upload GNN'} accept=".pkl" onFile={async file => {
                  const data = await uploadGnn(file);
                  setLoadedGnn({ id: data.gnn_id, name: file.name });
                }} />
                <UploadButton label={loadedPolicy ? `Policy: ${loadedPolicy.name}` : 'Upload Policy'} accept=".pkl" onFile={async file => {
                  const data = await uploadPolicy(file);
                  setLoadedPolicy({ id: data.policy_id, name: file.name, records: data.records_seen ?? 0 });
                }} />
                <UploadButton label={baseArch ? `Base: ${baseArch.name}` : 'Upload Base Arch'} accept=".pkl" onFile={async file => {
                  const data = await uploadArch(file);
                  setBaseArch({ id: data.arch_id, name: file.name });
                  setGraph(graphData(data));
                }} />
              </>
            )}
            <button className="btn" disabled={status === 'running'} onClick={start}>
              {status === 'running' ? 'Running...' : mode === 'train_encoder' ? 'Start Encoder Training' : 'Start Evolution'}
            </button>
          </div>

          {error && (
            <div style={{ marginTop: 14, color: '#ff6b6b', fontSize: 12, lineHeight: 1.5 }}>
              {error}
            </div>
          )}
        </aside>

        <main style={{ display: 'grid', gap: 14 }}>
          {mode === 'evolve' && (
            <section style={{ ...panel, padding: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, color: 'var(--text-primary)', fontWeight: 800 }}>
                <span>Evolution Epoch</span>
                <span>{epochProgress.current}/{epochProgress.total || epochsB}</span>
              </div>
              <div style={{ height: 7, background: 'rgba(148,163,184,0.12)', borderRadius: 99, overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${Math.min(100, Math.round((epochProgress.current / Math.max(epochProgress.total || epochsB, 1)) * 100))}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg,#38bdf8,var(--theme-accent))',
                  }}
                />
              </div>
            </section>
          )}
          <section style={{ ...panel, padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10, color: 'var(--text-primary)', fontWeight: 800 }}>
              <span>{mode === 'train_encoder' ? 'Encoder Tournament' : 'Evolution Tournament'}</span>
              <span>{fightProgress.current}/{fightProgress.total} fights · {progressPct}%</span>
            </div>
            <div style={{ height: 7, background: 'rgba(148,163,184,0.12)', borderRadius: 99, overflow: 'hidden', marginBottom: 12 }}>
              <div style={{ width: `${progressPct}%`, height: '100%', background: 'linear-gradient(90deg,var(--theme-primary),var(--theme-accent))' }} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, alignItems: 'start' }}>
              <Leaderboard archs={sortedArchs} maxHeight={mode === 'evolve' ? 150 : 250} />
              <FightStream fights={fightLog} archs={tournamentArchs} maxHeight={mode === 'evolve' ? 150 : 250} />
            </div>
          </section>

          <section style={{ display: 'grid', gridTemplateColumns: mode === 'evolve' ? '1fr' : '1fr 1fr', gap: 14 }}>
            <div style={{ ...panel, padding: 14, minHeight: 310 }}>
              <h3 style={{ margin: '0 0 10px', color: 'var(--text-primary)', fontSize: 15 }}>
                {mode === 'train_encoder' ? 'Average Prediction Error' : 'Predicted vs Arena Score'}
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={activeHistory}>
                  <CartesianGrid stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="epoch" stroke="#64748b" />
                  <YAxis stroke="#64748b" width={48} />
                  <Tooltip contentStyle={{ background: '#090604', border: '1px solid var(--glass-border)' }} />
                  {mode === 'train_encoder' ? (
                    <>
                      <Line type="monotone" dataKey="avg_delta" stroke="#62ff8a" dot={false} name="avg abs error" />
                      <Line type="monotone" dataKey="gnn_loss" stroke="#ffb000" dot={false} name="loss" />
                    </>
                  ) : (
                    <>
                      <Line type="monotone" dataKey="predicted_score" stroke="#38bdf8" dot={false} name="predicted" />
                      <Line type="monotone" dataKey="true_score" stroke="#62ff8a" dot={false} name="arena" />
                      <Line type="monotone" dataKey="best_score" stroke="#ffb000" dot={false} name="best" />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section style={{ display: 'grid', gridTemplateColumns: mode === 'evolve' ? '1fr 1fr' : '1fr', gap: 14 }}>
            <div style={{ ...panel, padding: 14, minHeight: 310 }}>
              <h3 style={{ margin: '0 0 10px', color: 'var(--text-primary)', fontSize: 15 }}>
                {mode === 'train_encoder' ? 'Best Architecture Seen' : 'Current Best / Living Architecture'}
              </h3>
              <div style={{ height: 260, border: '1px solid rgba(148,163,184,0.14)', borderRadius: 8, overflow: 'hidden' }}>
                <MiniArchFlow
                  id="rl-current-architecture"
                  nodes={graph.nodes}
                  edges={graph.edges}
                  backgroundColor="rgba(255,122,24,0.14)"
                  onNodesChange={onNodesChange}
                />
              </div>
            </div>

            {mode === 'evolve' && (
            <div style={{ ...panel, padding: 14, minHeight: 310 }}>
              <h3 style={{ margin: '0 0 10px', color: 'var(--text-primary)', fontSize: 15 }}>
                Architecture Currently Being Tested
              </h3>
              <div style={{ height: 260, border: '1px solid rgba(148,163,184,0.14)', borderRadius: 8, overflow: 'hidden' }}>
                <MiniArchFlow
                  id="rl-tested-architecture"
                  nodes={testedGraph.nodes}
                  edges={testedGraph.edges}
                  backgroundColor="rgba(56,189,248,0.14)"
                  onNodesChange={onTestedNodesChange}
                />
              </div>
            </div>
            )}
          </section>

          {mode === 'evolve' && (
            <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <VersionTable
                rows={versionRows}
                storedVersions={storedVersions}
                bestScore={bestScore}
                canRevert={status === 'done'}
                onSelectVersion={(version) => {
                  setFinalArchId(version.arch_id);
                  setGraph(graphData(version.graph));
                  setBestScore(version.score);
                }}
              />
              <InfoTable
                title={`Mutation Candidates${initialScore !== null ? ` · initial ${fmt(initialScore)}` : ''}`}
                headers={['Mutation', 'Pred Score', 'Lrn / Spd / Opp Raw', 'Nodes']}
                rows={candidateRows.map(r => [
                  r.mutation_sequence?.length ? r.mutation_sequence.join(' -> ') : `${r.mutation_type}${r.target_module_type ? ` ${r.target_module_type}` : ''}`,
                  fmt(r.predicted_score),
                  r.predicted_components ? `${fmt(r.predicted_components.learnability, 2)} / ${fmt(r.predicted_components.speed, 2)} / ${fmt(r.predicted_components.opp_simp_raw ?? r.predicted_components.opp_simp_bonus, 2)}` : '-',
                  String(r.n_nodes),
                ])}
              />
              <TopArchitecturesTable rows={topArchitectures} />
              <InfoTable
                title={`Architecture Versions${bestScore !== null ? ` · best ${fmt(bestScore)}` : ''}`}
                headers={['Version', 'Arena', 'Delta', 'Pred Err']}
                rows={versionRows.map(r => [
                  `${r.version} ${r.mutation_type}`,
                  fmt(r.arena_score),
                  fmt(r.arena_delta),
                  fmt(r.prediction_error),
                ])}
              />
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'grid', gap: 5 }}>
      <span style={{ color: 'var(--text-secondary)', fontSize: 11, fontWeight: 900, textTransform: 'uppercase' }}>{label}</span>
      {children}
    </label>
  );
}

function Leaderboard({ archs, maxHeight = 250 }: { archs: TournamentArch[]; maxHeight?: number }) {
  return (
    <div style={{ maxHeight, overflow: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ color: '#64748b', textAlign: 'left' }}>
            <th>Rank</th><th>Architecture</th><th>Score</th><th>Learn</th><th>Speed</th><th>Fights</th>
          </tr>
        </thead>
        <tbody>
          {archs.map((a, idx) => (
            <tr key={`${a.id}-${a.name}`} style={{ borderTop: '1px solid rgba(148,163,184,0.09)', color: 'var(--text-primary)' }}>
              <td>{idx + 1}</td>
              <td style={{ color: 'var(--theme-primary)' }}>{a.name}</td>
              <td>{fmt(a.score)}</td>
              <td>{fmt(a.learnability)}</td>
              <td>{fmt(a.speed)}</td>
              <td>{a.fight_count ?? 0}</td>
            </tr>
          ))}
          {archs.length === 0 && <tr><td colSpan={6} style={{ color: '#64748b', padding: 16 }}>Tournament architectures will appear here.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

function FightStream({ fights, archs, maxHeight = 250 }: { fights: FightLog[]; archs: TournamentArch[]; maxHeight?: number }) {
  return (
    <div style={{ maxHeight, overflow: 'auto', fontSize: 12, color: 'var(--text-primary)' }}>
      {fights.length === 0 && <div style={{ color: '#64748b' }}>Fight results will stream here.</div>}
      {fights.slice().reverse().map(f => (
        <div key={`${f.epoch}-${f.eval_role}-${f.fight}`} style={{ padding: '6px 0', borderBottom: '1px solid rgba(148,163,184,0.08)' }}>
          <strong style={{ color: f.failed ? '#ff6b6b' : 'var(--theme-accent)' }}>#{f.fight}/{f.total}</strong>{' '}
          {archs[f.i]?.name ?? f.i} vs {archs[f.j]?.name ?? f.j}{' '}
          <span style={{ color: '#94a3b8' }}>{fmt(f.score_i)} / {fmt(f.score_j)}</span>
        </div>
      ))}
    </div>
  );
}

function InfoTable({ title, headers, rows }: { title: string; headers: string[]; rows: string[][] }) {
  if (title.startsWith('Architecture Versions')) return null;
  return (
    <div style={{ ...panel, padding: 14, minHeight: 220 }}>
      <h3 style={{ margin: '0 0 10px', color: 'var(--text-primary)', fontSize: 15 }}>{title}</h3>
      <div style={{ maxHeight: 180, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr style={{ color: '#64748b', textAlign: 'left' }}>{headers.map(h => <th key={h}>{h}</th>)}</tr></thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx} style={{ borderTop: '1px solid rgba(148,163,184,0.09)', color: 'var(--text-primary)' }}>
                {row.map((cell, cidx) => <td key={cidx} style={{ padding: '6px 4px' }}>{cell}</td>)}
              </tr>
            ))}
            {rows.length === 0 && <tr><td colSpan={headers.length} style={{ color: '#64748b', padding: 16 }}>Waiting for data.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function topArchFilename(row: TopArchitecture) {
  const source = (row.source || `epoch ${row.epoch}`)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return `top_arch_${row.role}_${source}_rank_${row.rank}`;
}

function TopArchitecturesTable({ rows }: { rows: TopArchitecture[] }) {
  return (
    <div style={{ ...panel, padding: 14, minHeight: 240, gridColumn: '1 / -1' }}>
      <h3 style={{ margin: '0 0 10px', color: 'var(--text-primary)', fontSize: 15 }}>Top Architectures Met</h3>
      <div style={{ maxHeight: 220, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ color: '#64748b', textAlign: 'left' }}>
              <th>Rank</th><th>Type</th><th>Source</th><th>Score</th><th>Lrn / Spd / Opp Raw</th><th>Nodes</th><th>Params</th><th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={`${row.rank}-${row.arch_id}`} style={{ borderTop: '1px solid rgba(148,163,184,0.09)', color: 'var(--text-primary)' }}>
                <td style={{ padding: '6px 4px', color: 'var(--theme-accent)', fontWeight: 900 }}>#{row.rank}</td>
                <td style={{ color: row.role === 'evolved' ? '#62ff8a' : '#38bdf8', fontWeight: 800 }}>{row.role}</td>
                <td>{row.source}</td>
                <td>{fmt(row.score)}</td>
                <td>{`${fmt(row.learnability, 2)} / ${fmt(row.speed, 2)} / ${fmt(row.opp_simp_raw, 2)}`}</td>
                <td>{row.n_nodes}</td>
                <td>{row.n_params}</td>
                <td>
                  {row.arch_id && (
                    <button
                      className="btn btn-back"
                      style={{ fontSize: 11, padding: '3px 8px' }}
                      onClick={() => downloadArch(row.arch_id, topArchFilename(row))}
                    >
                      Download
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td colSpan={8} style={{ color: '#64748b', padding: 16 }}>Scored evolved and opponent architectures will appear here.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MiniArchFlow({
  id,
  nodes,
  edges,
  backgroundColor,
  onNodesChange,
}: {
  id: string;
  nodes: Node[];
  edges: Edge[];
  backgroundColor: string;
  onNodesChange?: OnNodesChange;
}) {
  return (
    <ReactFlowProvider>
      <ReactFlow
        id={id}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        fitView
        minZoom={0.2}
        maxZoom={3}
        proOptions={{ hideAttribution: true }}
      >
        <Background color={backgroundColor} gap={18} />
        <Controls />
      </ReactFlow>
    </ReactFlowProvider>
  );
}

function VersionTable({
  rows,
  storedVersions,
  bestScore,
  canRevert,
  onSelectVersion,
}: {
  rows: VersionRow[];
  storedVersions: StoredVersion[];
  bestScore: number | null;
  canRevert: boolean;
  onSelectVersion: (version: StoredVersion) => void;
}) {
  const byVersion = new Map(storedVersions.map(v => [v.version, v]));
  return (
    <div style={{ ...panel, padding: 14, minHeight: 220 }}>
      <h3 style={{ margin: '0 0 10px', color: 'var(--text-primary)', fontSize: 15 }}>
        Restorable Versions{bestScore !== null ? ` - best ${fmt(bestScore)}` : ''}
      </h3>
      <div style={{ maxHeight: 180, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ color: '#64748b', textAlign: 'left' }}>
              <th>Version</th><th>Arena</th><th>Delta</th><th>Lrn / Spd / Opp Raw</th><th>Status</th><th></th>
            </tr>
          </thead>
          <tbody>
            {storedVersions.map(v => {
              const components = [v.true_learnability, v.true_speed, v.true_opp_simp_raw_bonus ?? v.true_opp_simp_bonus];
              const status = v.version === 0
                ? 'initial'
                : v.accepted
                  ? `accepted p=${fmt(v.acceptance_probability, 2)}`
                  : v.exploratory
                    ? `explored p=${fmt(v.acceptance_probability, 2)}`
                    : `rejected p=${fmt(v.acceptance_probability, 2)}`;
              return (
                <tr key={`stored-${v.version}`} style={{ borderTop: '1px solid rgba(148,163,184,0.09)', color: 'var(--text-primary)' }}>
                  <td>v{v.version}</td>
                  <td>{fmt(v.score)}</td>
                  <td style={{ color: (v.arena_delta ?? 0) > 0 ? '#62ff8a' : (v.arena_delta ?? 0) < 0 ? '#ff6b6b' : 'var(--text-muted)' }}>
                    {v.version === 0 ? '-' : fmt(v.arena_delta)}
                  </td>
                  <td>{components.some(value => value != null) ? `${fmt(components[0], 2)} / ${fmt(components[1], 2)} / ${fmt(components[2], 2)}` : '-'}</td>
                  <td>{status}</td>
                  <td>
                    {canRevert && (
                      <button className="btn btn-back" style={{ fontSize: 11, padding: '3px 8px' }} onClick={() => onSelectVersion(v)}>
                        Revert
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
            {rows.map((r, idx) => {
              const stored = r.version !== null ? byVersion.get(r.version) : undefined;
              if (stored) return null;
              return (
                <tr key={idx} style={{ borderTop: '1px solid rgba(148,163,184,0.09)', color: 'var(--text-primary)' }}>
                  <td style={{ padding: '6px 4px' }}>{r.version !== null ? `v${r.version}` : `epoch ${r.epoch ?? idx + 1}`}</td>
                  <td>{fmt(r.arena_score)}</td>
                  <td style={{ color: r.arena_delta > 0 ? '#62ff8a' : '#ff6b6b' }}>{fmt(r.arena_delta)}</td>
                  <td>{`${fmt(r.true_learnability, 2)} / ${fmt(r.true_speed, 2)} / ${fmt(r.true_opp_simp_raw_bonus ?? r.true_opp_simp_bonus, 2)}`}</td>
                  <td>{r.accepted ? `accepted p=${fmt(r.acceptance_probability, 2)}` : r.exploratory ? `explored p=${fmt(r.acceptance_probability, 2)}` : 'rejected'}</td>
                  <td>
                    {canRevert && stored && (
                      <button className="btn btn-back" style={{ fontSize: 11, padding: '3px 8px' }} onClick={() => onSelectVersion(stored)}>
                        Revert
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && storedVersions.length === 0 && (
              <tr><td colSpan={6} style={{ color: '#64748b', padding: 16 }}>Waiting for versions.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function RlSearchViewer({ initialMode = 'train_encoder' }: { initialMode?: Mode }) {
  return <RlSearchViewerInner initialMode={initialMode} />;
}
