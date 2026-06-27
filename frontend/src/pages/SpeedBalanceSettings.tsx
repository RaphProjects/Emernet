import { useEffect, useState } from 'react';
import type React from 'react';
import { getClientId, getSpeedBalance, updateSpeedBalance } from '../api';
import { INPUT_STYLE, MODULE_SET_OPTIONS } from '../theme';
import LooseNumberInput from '../components/LooseNumberInput';

type CalibrationRound = {
  tournament: number;
  speed_bal: number;
  score_spread: number;
  mlps: { name: string; learnability: number; speed: number; score: number }[];
};

type CalibrationResult = {
  module_group: string;
  n_tournaments: number;
  speed_bal: number;
  std: number;
  rounds: CalibrationRound[];
};

const panel: React.CSSProperties = {
  background: 'var(--glass-bg)',
  border: '1px solid var(--glass-border)',
  borderRadius: 8,
  boxShadow: 'var(--window-shadow)',
};

const input = { ...INPUT_STYLE, width: '100%' };

export default function SpeedBalanceSettings({ embedded = false }: { embedded?: boolean } = {}) {
  const [values, setValues] = useState<Record<string, number> | null>(null);
  const [status, setStatus] = useState('Loading speed balance values...');
  const [saving, setSaving] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState('Idle.');
  const [calibrationProgress, setCalibrationProgress] = useState({ current: 0, total: 0 });
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResult | null>(null);
  const [nTournaments, setNTournaments] = useState(3);
  const [poolSize, setPoolSize] = useState(8);
  const [archSize, setArchSize] = useState(12);
  const [moduleSet, setModuleSet] = useState('Unified');
  const [datasetSize, setDatasetSize] = useState(320);
  const [nFights, setNFights] = useState(1);
  const [maxDuration, setMaxDuration] = useState(600);

  useEffect(() => {
    getSpeedBalance()
      .then(data => {
        setValues(data.speed_balance);
        setStatus('Loaded from backend');
      })
      .catch(err => setStatus(err.message));
  }, []);

  const update = (group: string, value: number) => {
    setValues(prev => prev ? ({ ...prev, [group]: clamp01(value) }) : prev);
  };

  const save = async () => {
    if (!values) return;
    setSaving(true);
    try {
      const data = await updateSpeedBalance(values);
      setValues(data.speed_balance);
      setStatus('Saved. New arena runs will use these speed balances.');
    } catch (err: any) {
      setStatus(err.message ?? 'Could not save speed balance values.');
    } finally {
      setSaving(false);
    }
  };

  const startCalibration = () => {
    setCalibrating(true);
    setCalibrationResult(null);
    setCalibrationProgress({ current: 0, total: 0 });
    setCalibrationStatus('Connecting...');
    const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE_URL}/ws/speed_balance_calibration`);
    ws.onopen = () => {
      ws.send(JSON.stringify({
        n_tournaments: nTournaments,
        pool_size: poolSize,
        module_set: moduleSet,
        client_id: getClientId(),
        arch_size: archSize,
        dataset_size: datasetSize,
        n_fights: nFights,
        max_duration: maxDuration,
      }));
    };
    ws.onmessage = event => {
      const data = JSON.parse(event.data);
      if (data.type === 'speed_balance_start') {
        setCalibrationProgress({ current: 0, total: data.total_fights ?? 0 });
        setCalibrationStatus(`Running ${data.n_tournaments} calibration round(s), ${data.opponent_pool_size} opponents per MLP.`);
      } else if (data.type === 'speed_balance_generation') {
        setCalibrationStatus(`Round ${data.tournament}: generating shared opponents ${data.current}/${data.total}.`);
      } else if (data.type === 'speed_balance_fight') {
        setCalibrationProgress({ current: data.fight ?? 0, total: data.total_fights ?? 0 });
        setCalibrationStatus(`Round ${data.tournament}: ${data.mlp} vs opponent ${data.opponent + 1}, fight ${data.fight}/${data.total_fights}.`);
      } else if (data.type === 'speed_balance_tournament_done') {
        setCalibrationStatus(`Tournament ${data.result.tournament} done: speed_bal ${fmt(data.result.speed_bal)}, spread ${fmt(data.result.score_spread)}.`);
      } else if (data.type === 'speed_balance_done') {
        setCalibrationResult(data.result);
        setCalibrationStatus(`Done. Mean speed_bal ${fmt(data.result.speed_bal)} +/- ${fmt(data.result.std)}.`);
        setCalibrating(false);
        ws.close();
      } else if (data.type === 'error') {
        setCalibrationStatus(data.message ?? 'Calibration failed.');
        setCalibrating(false);
        ws.close();
      }
    };
    ws.onerror = () => {
      setCalibrationStatus('Connection error. Is the backend running?');
      setCalibrating(false);
    };
    ws.onclose = () => setCalibrating(false);
  };

  const applyCalibration = () => {
    if (!calibrationResult) return;
    setValues(prev => prev ? ({
      ...prev,
      [calibrationResult.module_group]: calibrationResult.speed_bal,
    }) : prev);
    setStatus('Calibration value staged. Click Save Values to persist it.');
  };

  return (
    <div style={{ flex: embedded ? undefined : 1, overflowY: embedded ? 'visible' : 'auto', height: embedded ? undefined : '100%' }}>
      <div style={{ maxWidth: embedded ? 'none' : 980, margin: '0 auto', padding: embedded ? 0 : '32px 20px' }}>
        <h1 style={{ color: 'var(--text-primary)', fontSize: 28, margin: '0 0 8px' }}>Speed Balance</h1>
        <p style={{ color: 'var(--text-secondary)', margin: '0 0 22px' }}>
          Calibrate the blend between normalized learnability and normalized speed.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14 }}>
          {values && Object.entries(values).map(([group, value]) => (
            <section key={group} style={{ ...panel, padding: 18 }}>
              <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 14px' }}>{group}</h2>
              <Field label="speed_bal">
                <LooseNumberInput
                  style={input}
                  min={0}
                  max={1}
                  step="0.0001"
                  value={value}
                  onChange={next => update(group, next)}
                />
              </Field>
            </section>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 16 }}>
          <button className="btn" disabled={!values || saving} onClick={save}>
            {saving ? 'Saving...' : 'Save Values'}
          </button>
          <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{status}</span>
        </div>

        <section style={{ ...panel, padding: 18, marginTop: 18 }}>
          <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 14px' }}>Run MLP Size Calibration</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
            <Field label="Calibration Rounds">
              <LooseNumberInput style={input} min={1} value={nTournaments} onChange={setNTournaments} fallback={1} />
            </Field>
            <Field label="Opponents / MLP">
              <LooseNumberInput style={input} min={2} value={poolSize} onChange={setPoolSize} fallback={2} />
            </Field>
            <Field label="Arch Size">
              <LooseNumberInput style={input} min={3} value={archSize} onChange={setArchSize} fallback={3} />
            </Field>
            <Field label="Dataset Size">
              <LooseNumberInput style={input} min={16} value={datasetSize} onChange={setDatasetSize} fallback={16} />
            </Field>
            <Field label="Arena Fights">
              <LooseNumberInput style={input} min={1} value={nFights} onChange={setNFights} fallback={1} />
            </Field>
            <Field label="Fight Timeout">
              <LooseNumberInput style={input} min={1} value={maxDuration} onChange={setMaxDuration} fallback={1} />
            </Field>
            <Field label="Module Set">
              <select style={input} value={moduleSet} onChange={e => setModuleSet(e.target.value)}>
                {MODULE_SET_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
              </select>
            </Field>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 14, flexWrap: 'wrap' }}>
            <button className="btn" disabled={calibrating} onClick={startCalibration}>
              {calibrating ? 'Running...' : 'Run Calibration'}
            </button>
            {calibrationResult && (
              <button className="btn btn-apply" onClick={applyCalibration}>
                Stage Speed Balance
              </button>
            )}
            <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{calibrationStatus}</span>
          </div>
          <div style={{ height: 7, background: 'rgba(148,163,184,0.12)', borderRadius: 99, overflow: 'hidden', marginTop: 12 }}>
            <div style={{ width: `${calibrationProgress.total ? Math.round((calibrationProgress.current / calibrationProgress.total) * 100) : 0}%`, height: '100%', background: 'linear-gradient(90deg,var(--theme-primary),var(--theme-accent))' }} />
          </div>

          {calibrationResult && (
            <div style={{ marginTop: 14, display: 'grid', gap: 14 }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                Suggested {calibrationResult.module_group} speed_bal: <strong style={{ color: 'var(--text-primary)' }}>{fmt(calibrationResult.speed_bal)}</strong>
                <span style={{ color: 'var(--text-muted)' }}> +/- {fmt(calibrationResult.std)}</span>
              </div>
              {calibrationResult.rounds.map(round => (
                <RoundTable key={round.tournament} round={round} />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function RoundTable({ round }: { round: CalibrationRound }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ color: 'var(--text-muted)', fontSize: 12, marginBottom: 6 }}>
        Tournament {round.tournament} · speed_bal {fmt(round.speed_bal)} · spread {fmt(round.score_spread)}
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-secondary)', fontSize: 12 }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
            <th style={{ padding: 8 }}>MLP</th>
            <th style={{ padding: 8 }}>Learnability</th>
            <th style={{ padding: 8 }}>Speed</th>
            <th style={{ padding: 8 }}>Final At Optimum</th>
          </tr>
        </thead>
        <tbody>
          {round.mlps.map(row => (
            <tr key={row.name} style={{ borderTop: '1px solid var(--glass-border)' }}>
              <td style={{ padding: 8 }}>{row.name}</td>
              <td style={{ padding: 8 }}>{fmt(row.learnability)}</td>
              <td style={{ padding: 8 }}>{fmt(row.speed)}</td>
              <td style={{ padding: 8 }}>{fmt(row.score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'grid', gap: 5, color: 'var(--text-secondary)', fontSize: 12 }}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function fmt(value: any, digits = 4) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
}
