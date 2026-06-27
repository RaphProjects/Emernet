import { useEffect, useState } from 'react';
import type React from 'react';
import { getArenaNormalization, getClientId, updateArenaNormalization } from '../api';
import { INPUT_STYLE, MODULE_SET_OPTIONS } from '../theme';
import LooseNumberInput from '../components/LooseNumberInput';

type NormGroup = {
  avg_learn: number;
  std_learn: number;
  avg_speed: number;
  std_speed: number;
  avg_simp?: number;
  std_simp?: number;
};

type CalibrationResult = {
  module_group: string;
  n_samples: number;
  learnability: { mean: number; std: number };
  speed: { mean: number; std: number };
  opp_simp_raw: { mean: number; std: number };
};

const panel: React.CSSProperties = {
  background: 'var(--glass-bg)',
  border: '1px solid var(--glass-border)',
  borderRadius: 8,
  boxShadow: 'var(--window-shadow)',
};

const input = { ...INPUT_STYLE, width: '100%' };

export default function NormalizationSettings({ embedded = false }: { embedded?: boolean } = {}) {
  const [values, setValues] = useState<Record<string, NormGroup> | null>(null);
  const [status, setStatus] = useState('Loading normalization values...');
  const [saving, setSaving] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState('Idle.');
  const [calibrationProgress, setCalibrationProgress] = useState({ current: 0, total: 0 });
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResult | null>(null);
  const [nTournaments, setNTournaments] = useState(3);
  const [nRandom, setNRandom] = useState(8);
  const [archSize, setArchSize] = useState(12);
  const [moduleSet, setModuleSet] = useState('Unified');
  const [datasetSize, setDatasetSize] = useState(320);
  const [nFights, setNFights] = useState(1);
  const [maxDuration, setMaxDuration] = useState(600);

  useEffect(() => {
    getArenaNormalization()
      .then(data => {
        setValues(data.normalization);
        setStatus('Loaded from backend');
      })
      .catch(err => setStatus(err.message));
  }, []);

  const update = (group: string, field: keyof NormGroup, value: number) => {
    setValues(prev => prev ? ({
      ...prev,
      [group]: { ...prev[group], [field]: value },
    }) : prev);
  };

  const startCalibration = () => {
    setCalibrating(true);
    setCalibrationResult(null);
    setCalibrationProgress({ current: 0, total: 0 });
    setCalibrationStatus('Connecting...');
    const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE_URL}/ws/normalization_calibration`);
    ws.onopen = () => {
      ws.send(JSON.stringify({
        n_tournaments: nTournaments,
        n_random: nRandom,
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
      if (data.type === 'calibration_start') {
        setCalibrationProgress({ current: 0, total: data.total_fights ?? 0 });
        setCalibrationStatus(`Running ${data.n_tournaments} tournament(s), ${data.n_random} architectures each.`);
      } else if (data.type === 'generation_progress') {
        setCalibrationStatus(`Tournament ${data.tournament}: generating ${data.current}/${data.total}.`);
      } else if (data.type === 'calibration_fight') {
        setCalibrationProgress({ current: data.fight ?? 0, total: data.total_fights ?? 0 });
        setCalibrationStatus(`Tournament ${data.tournament}: fight ${data.fight}/${data.total_fights}.`);
      } else if (data.type === 'calibration_tournament_done') {
        setCalibrationStatus(`Tournament ${data.tournament} done: learn ${fmt(data.learn_mean)}, speed ${fmt(data.speed_mean)}, opp ${fmt(data.simp_mean)}.`);
      } else if (data.type === 'calibration_done') {
        setCalibrationResult(data.result);
        setCalibrationStatus(`Done. ${data.result.n_samples} architecture samples collected.`);
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
    const group = calibrationResult.module_group;
    setValues(prev => prev ? ({
      ...prev,
      [group]: {
        ...prev[group],
        avg_learn: calibrationResult.learnability.mean,
        std_learn: calibrationResult.learnability.std,
        avg_speed: calibrationResult.speed.mean,
        std_speed: calibrationResult.speed.std,
        avg_simp: calibrationResult.opp_simp_raw.mean,
        std_simp: calibrationResult.opp_simp_raw.std,
      },
    }) : prev);
    setStatus('Calibration values staged. Click Save Values to persist them.');
  };

  const save = async () => {
    if (!values) return;
    setSaving(true);
    try {
      const data = await updateArenaNormalization(values);
      setValues(data.normalization);
      setStatus('Saved. New arena runs will use these values.');
    } catch (err: any) {
      setStatus(err.message ?? 'Could not save normalization values.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ flex: embedded ? undefined : 1, overflowY: embedded ? 'visible' : 'auto', height: embedded ? undefined : '100%' }}>
      <div style={{ maxWidth: embedded ? 'none' : 920, margin: '0 auto', padding: embedded ? 0 : '32px 20px' }}>
        <h1 style={{ color: 'var(--text-primary)', fontSize: 28, margin: '0 0 8px' }}>Arena Normalization</h1>
        <p style={{ color: 'var(--text-secondary)', margin: '0 0 22px' }}>
          Learnability and speed are normalized before arena scores are combined.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
          {values && Object.entries(values).map(([group, norm]) => (
            <section key={group} style={{ ...panel, padding: 18 }}>
              <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 14px' }}>
                {group === 'default' ? 'Rich / All' : group}
              </h2>
              <div style={{ display: 'grid', gap: 12 }}>
                {(['avg_learn', 'std_learn', 'avg_speed', 'std_speed', 'avg_simp', 'std_simp'] as const).map(field => (
                  <label key={field} style={{ display: 'grid', gap: 5, color: 'var(--text-secondary)', fontSize: 12 }}>
                    <span>{field}</span>
                    <LooseNumberInput
                      style={input}
                      step="0.0001"
                      value={norm[field] ?? (field === 'avg_simp' ? norm.avg_learn : norm.std_learn)}
                      onChange={value => update(group, field, value)}
                    />
                  </label>
                ))}
              </div>
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
          <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 14px' }}>Run Calibration Tournaments</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
            <Field label="Tournaments">
              <LooseNumberInput style={input} min={1} value={nTournaments} onChange={setNTournaments} fallback={1} />
            </Field>
            <Field label="Random Archs">
              <LooseNumberInput style={input} min={2} value={nRandom} onChange={setNRandom} fallback={2} />
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
                Stage Calibration Values
              </button>
            )}
            <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{calibrationStatus}</span>
          </div>
          <div style={{ height: 7, background: 'rgba(148,163,184,0.12)', borderRadius: 99, overflow: 'hidden', marginTop: 12 }}>
            <div style={{ width: `${calibrationProgress.total ? Math.round((calibrationProgress.current / calibrationProgress.total) * 100) : 0}%`, height: '100%', background: 'linear-gradient(90deg,var(--theme-primary),var(--theme-accent))' }} />
          </div>
          {calibrationResult && (
            <div style={{ marginTop: 14, overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-secondary)', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                    <th style={{ padding: 8 }}>Metric</th>
                    <th style={{ padding: 8 }}>Mean</th>
                    <th style={{ padding: 8 }}>Std</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['learnability', calibrationResult.learnability],
                    ['speed', calibrationResult.speed],
                    ['raw opp simp', calibrationResult.opp_simp_raw],
                  ].map(([label, metric]: any) => (
                    <tr key={label} style={{ borderTop: '1px solid var(--glass-border)' }}>
                      <td style={{ padding: 8 }}>{label}</td>
                      <td style={{ padding: 8 }}>{fmt(metric.mean)}</td>
                      <td style={{ padding: 8 }}>{fmt(metric.std)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
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

function fmt(value: any, digits = 4) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
}
