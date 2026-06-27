import { useState } from 'react';
import type React from 'react';
import { getClientId, getPolicyInfo, uploadPolicy } from '../api';
import { INPUT_STYLE, MODULE_SET_OPTIONS } from '../theme';
import LooseNumberInput from '../components/LooseNumberInput';

const panel: React.CSSProperties = {
  background: 'var(--glass-bg)',
  border: '1px solid var(--glass-border)',
  borderRadius: 8,
  boxShadow: 'var(--window-shadow)',
};

function fmt(value: any, digits = 4) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : String(value ?? '-');
}

export default function PolicyInspector() {
  const [info, setInfo] = useState<any>(null);
  const [policyId, setPolicyId] = useState<string | null>(null);
  const [status, setStatus] = useState('Upload an evolution policy .pkl to inspect it.');
  const [loading, setLoading] = useState(false);
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalStatus, setEvalStatus] = useState('Idle.');
  const [evalProgress, setEvalProgress] = useState({ current: 0, total: 0 });
  const [evalResult, setEvalResult] = useState<any>(null);
  const [evalTournaments, setEvalTournaments] = useState(3);
  const [evalTournamentSize, setEvalTournamentSize] = useState(8);
  const [evalArchSize, setEvalArchSize] = useState(12);
  const [evalDatasetSize, setEvalDatasetSize] = useState(320);
  const [evalFights, setEvalFights] = useState(1);
  const [evalMaxDuration, setEvalMaxDuration] = useState(600);
  const [evalSimpOppBal, setEvalSimpOppBal] = useState(0.2);
  const [evalModuleSet, setEvalModuleSet] = useState('Unified');

  const load = async (file: File) => {
    setLoading(true);
    try {
      const uploaded = await uploadPolicy(file);
      const data = await getPolicyInfo(uploaded.policy_id);
      setPolicyId(uploaded.policy_id);
      setInfo({ ...data, filename: file.name });
      setEvalResult(null);
      setStatus('Policy loaded.');
    } catch (err: any) {
      setStatus(err.message ?? 'Could not inspect policy.');
    } finally {
      setLoading(false);
    }
  };

  const runEvaluation = () => {
    if (!policyId || evalRunning) return;
    setEvalRunning(true);
    setEvalResult(null);
    setEvalProgress({ current: 0, total: 0 });
    setEvalStatus('Connecting...');
    const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE_URL}/ws/policy_eval`);
    ws.onopen = () => {
      ws.send(JSON.stringify({
        policy_id: policyId,
        n_tournaments: evalTournaments,
        tournament_size: evalTournamentSize,
        module_set: evalModuleSet,
        arch_size: evalArchSize,
        dataset_size: evalDatasetSize,
        n_fights: evalFights,
        max_duration: evalMaxDuration,
        simp_opp_bal: evalSimpOppBal,
        client_id: getClientId(),
      }));
    };
    ws.onmessage = event => {
      const data = JSON.parse(event.data);
      if (data.type === 'policy_eval_start') {
        setEvalProgress({ current: 0, total: data.total_fights ?? 0 });
        setEvalStatus(`Running ${data.n_tournaments} tournament(s), ${data.tournament_size} architectures each.`);
      } else if (data.type === 'policy_eval_generation') {
        setEvalStatus(`Tournament ${data.tournament}: generating ${data.current}/${data.total}.`);
      } else if (data.type === 'policy_eval_predictions') {
        setEvalStatus(`Tournament ${data.tournament}: policy predictions computed.`);
      } else if (data.type === 'policy_eval_fight') {
        setEvalProgress(prev => ({ current: prev.current + 1, total: prev.total || data.total || 0 }));
        setEvalStatus(`Tournament ${data.tournament}: fight ${data.fight}/${data.total}.`);
      } else if (data.type === 'policy_eval_tournament_done') {
        setEvalStatus(`Tournament ${data.tournament} done: average delta ${fmt(data.metrics?.mae)}.`);
      } else if (data.type === 'policy_eval_done') {
        setEvalResult(data.result);
        setEvalStatus(`Done. ${data.result.n_samples} predictions evaluated.`);
        setEvalRunning(false);
        ws.close();
      } else if (data.type === 'error') {
        setEvalStatus(data.message ?? 'Policy evaluation failed.');
        setEvalRunning(false);
        ws.close();
      }
    };
    ws.onerror = () => {
      setEvalStatus('Connection error. Is the backend running?');
      setEvalRunning(false);
    };
    ws.onclose = () => setEvalRunning(false);
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto', height: '100%' }}>
      <div style={{ maxWidth: 980, margin: '0 auto', padding: '32px 20px' }}>
        <h1 style={{ color: 'var(--text-primary)', fontSize: 28, margin: '0 0 8px' }}>Policy Inspector</h1>
        <p style={{ color: 'var(--text-secondary)', margin: '0 0 18px' }}>
          Inspect replay size and the hybrid model weights stored in an evolution policy.
        </p>

        <section style={{ ...panel, padding: 16, marginBottom: 14 }}>
          <label className="btn" style={{ display: 'inline-flex', cursor: loading ? 'not-allowed' : 'pointer' }}>
            {loading ? 'Loading...' : 'Upload Policy'}
            <input
              type="file"
              accept=".pkl"
              disabled={loading}
              onChange={e => {
                const file = e.target.files?.[0];
                if (file) load(file);
                e.currentTarget.value = '';
              }}
              style={{ display: 'none' }}
            />
          </label>
          <span style={{ color: 'var(--text-muted)', fontSize: 12, marginLeft: 12 }}>{status}</span>
        </section>

        {info && (
          <div style={{ display: 'grid', gap: 14 }}>
            <section style={{ ...panel, padding: 16 }}>
              <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 12px' }}>{info.filename}</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
                <Metric label="Replay Records" value={info.replay_records} />
                <Metric label="Records Seen" value={info.records_seen} />
                <Metric label="Feature Dim" value={info.feature_dim} />
                <Metric label="LGBM" value={info.has_lgbm ? 'trained' : 'not trained'} />
                <Metric label="NN" value={info.has_nn ? 'trained' : 'not trained'} />
                <Metric label="Meta" value={info.has_meta ? 'trained' : 'not trained'} />
              </div>
            </section>

            <section style={{ ...panel, padding: 16 }}>
              <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 12px' }}>Meta Layer</h2>
              {info.meta?.weights?.length ? (
                <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-secondary)', fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                      <th style={{ padding: 8 }}>Metric</th>
                      <th style={{ padding: 8 }}>Input</th>
                      <th style={{ padding: 8 }}>Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {info.meta.weights.map((row: any) => (
                      <tr key={`${row.metric}-${row.name}`} style={{ borderTop: '1px solid var(--glass-border)' }}>
                        <td style={{ padding: 8 }}>{row.metric ?? '-'}</td>
                        <td style={{ padding: 8 }}>{row.name}</td>
                        <td style={{ padding: 8, color: row.value >= 0 ? '#62ff8a' : '#ff6b6b' }}>{fmt(row.value)}</td>
                      </tr>
                    ))}
                    {(Array.isArray(info.meta.bias) ? info.meta.bias : [info.meta.bias]).map((value: any, idx: number) => (
                      <tr key={`bias-${idx}`} style={{ borderTop: '1px solid var(--glass-border)' }}>
                        <td style={{ padding: 8 }}>{['learnability', 'speed', 'opp_raw'][idx] ?? `metric_${idx}`}</td>
                        <td style={{ padding: 8 }}>bias</td>
                        <td style={{ padding: 8 }}>{fmt(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p style={{ color: 'var(--text-muted)', margin: 0 }}>No trained meta layer weights found.</p>
              )}
            </section>

            <section style={{ ...panel, padding: 16 }}>
              <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 12px' }}>Metric Heads</h2>
              <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-secondary)', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                    <th style={{ padding: 8 }}>Metric</th>
                    <th style={{ padding: 8 }}>LGBM</th>
                    <th style={{ padding: 8 }}>NN</th>
                    <th style={{ padding: 8 }}>Meta</th>
                  </tr>
                </thead>
                <tbody>
                  {(info.metrics ?? []).map((row: any) => (
                    <tr key={row.metric} style={{ borderTop: '1px solid var(--glass-border)' }}>
                      <td style={{ padding: 8 }}>{row.metric}</td>
                      <td style={{ padding: 8 }}>{row.lgbm}</td>
                      <td style={{ padding: 8 }}>{row.nn}</td>
                      <td style={{ padding: 8 }}>{row.meta}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section style={{ ...panel, padding: 16 }}>
              <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 12px' }}>Prediction Quality</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
                <Field label="Tournaments">
                  <LooseNumberInput style={input} min={1} value={evalTournaments} onChange={setEvalTournaments} fallback={1} />
                </Field>
                <Field label="Tournament Size">
                  <LooseNumberInput style={input} min={2} max={64} value={evalTournamentSize} onChange={setEvalTournamentSize} fallback={2} />
                </Field>
                <Field label="Arch Size">
                  <LooseNumberInput style={input} min={3} max={64} value={evalArchSize} onChange={setEvalArchSize} fallback={3} />
                </Field>
                <Field label="Dataset Size">
                  <LooseNumberInput style={input} min={16} value={evalDatasetSize} onChange={setEvalDatasetSize} fallback={16} />
                </Field>
                <Field label="Arena Fights">
                  <LooseNumberInput style={input} min={1} value={evalFights} onChange={setEvalFights} fallback={1} />
                </Field>
                <Field label="Fight Timeout">
                  <LooseNumberInput style={input} min={1} value={evalMaxDuration} onChange={setEvalMaxDuration} fallback={1} />
                </Field>
                <Field label="Opp Simp Bal">
                  <LooseNumberInput style={input} step="0.05" value={evalSimpOppBal} onChange={setEvalSimpOppBal} />
                </Field>
                <Field label="Module Set">
                  <select style={input} value={evalModuleSet} onChange={e => setEvalModuleSet(e.target.value)}>
                    {MODULE_SET_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                  </select>
                </Field>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 14, flexWrap: 'wrap' }}>
                <button className="btn" disabled={!policyId || evalRunning} onClick={runEvaluation}>
                  {evalRunning ? 'Evaluating...' : 'Run Prediction Tournaments'}
                </button>
                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{evalStatus}</span>
              </div>
              <div style={{ height: 7, background: 'rgba(148,163,184,0.12)', borderRadius: 99, overflow: 'hidden', marginTop: 12 }}>
                <div style={{
                  width: `${evalProgress.total ? Math.min(100, Math.round((evalProgress.current / evalProgress.total) * 100)) : 0}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg,var(--theme-primary),var(--theme-accent))',
                }} />
              </div>

              {evalResult && (
                <div style={{ display: 'grid', gap: 14, marginTop: 14 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                    <Metric label="Average Delta" value={fmt(evalResult.metrics?.mae)} />
                    <Metric label="RMSE" value={fmt(evalResult.metrics?.rmse)} />
                    <Metric label="Bias" value={fmt(evalResult.metrics?.bias)} />
                    <Metric label="Max Delta" value={fmt(evalResult.metrics?.max_abs)} />
                    <Metric label="Correlation" value={fmt(evalResult.metrics?.corr)} />
                    <Metric label="Samples" value={evalResult.n_samples} />
                  </div>

                  <ComponentMetrics metrics={evalResult.component_metrics} />
                  <PredictionRows rows={evalResult.rows ?? []} />
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}

const input = { ...INPUT_STYLE, width: '100%' };

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'grid', gap: 5, color: 'var(--text-secondary)', fontSize: 12 }}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function ComponentMetrics({ metrics }: { metrics: Record<string, any> | undefined }) {
  if (!metrics) return null;
  const labels: Record<string, string> = {
    learnability: 'Learnability',
    speed: 'Speed',
    opp_simp_raw: 'Opp Raw',
  };
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-secondary)', fontSize: 12 }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
            <th style={{ padding: 8 }}>Component</th>
            <th style={{ padding: 8 }}>Average Delta</th>
            <th style={{ padding: 8 }}>RMSE</th>
            <th style={{ padding: 8 }}>Bias</th>
            <th style={{ padding: 8 }}>Correlation</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([key, row]: any) => (
            <tr key={key} style={{ borderTop: '1px solid var(--glass-border)' }}>
              <td style={{ padding: 8 }}>{labels[key] ?? key}</td>
              <td style={{ padding: 8 }}>{fmt(row.mae)}</td>
              <td style={{ padding: 8 }}>{fmt(row.rmse)}</td>
              <td style={{ padding: 8 }}>{fmt(row.bias)}</td>
              <td style={{ padding: 8 }}>{fmt(row.corr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PredictionRows({ rows }: { rows: any[] }) {
  const sorted = [...rows].sort((a, b) => (b.abs_delta ?? 0) - (a.abs_delta ?? 0));
  return (
    <div style={{ overflowX: 'auto', maxHeight: 360, overflowY: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-secondary)', fontSize: 12, minWidth: 820 }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', textAlign: 'left', position: 'sticky', top: 0, background: 'rgba(0,0,0,0.82)' }}>
            <th style={{ padding: 8 }}>Tournament</th>
            <th style={{ padding: 8 }}>Arch</th>
            <th style={{ padding: 8 }}>Pred</th>
            <th style={{ padding: 8 }}>Arena</th>
            <th style={{ padding: 8 }}>Delta</th>
            <th style={{ padding: 8 }}>Pred Lrn / Spd / Opp</th>
            <th style={{ padding: 8 }}>Arena Lrn / Spd / Opp</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, idx) => (
            <tr key={`${row.tournament}-${row.arch}-${idx}`} style={{ borderTop: '1px solid var(--glass-border)' }}>
              <td style={{ padding: 8 }}>{row.tournament}</td>
              <td style={{ padding: 8 }}>{row.arch}</td>
              <td style={{ padding: 8 }}>{fmt(row.predicted)}</td>
              <td style={{ padding: 8 }}>{fmt(row.actual)}</td>
              <td style={{ padding: 8, color: Math.abs(row.delta ?? 0) < 0.1 ? '#62ff8a' : '#ffb000' }}>{fmt(row.delta)}</td>
              <td style={{ padding: 8 }}>
                {fmt(row.predicted_learnability, 2)} / {fmt(row.predicted_speed, 2)} / {fmt(row.predicted_opp_simp_raw, 2)}
              </td>
              <td style={{ padding: 8 }}>
                {fmt(row.actual_learnability, 2)} / {fmt(row.actual_speed, 2)} / {fmt(row.actual_opp_simp_raw, 2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: any }) {
  return (
    <div style={{ border: '1px solid var(--glass-border)', borderRadius: 8, padding: 10 }}>
      <div style={{ color: 'var(--text-muted)', fontSize: 11, marginBottom: 4 }}>{label}</div>
      <div style={{ color: 'var(--text-primary)', fontWeight: 800 }}>{String(value ?? '-')}</div>
    </div>
  );
}
