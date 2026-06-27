import { useState, useEffect, useRef } from 'react';
import type React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts';
import { MODULE_SET_OPTIONS, INPUT_STYLE } from '../theme';
import { downloadArch, getClientId } from '../api';
import LooseNumberInput from '../components/LooseNumberInput';

const API = import.meta.env.VITE_API_BASE_URL;

//  types 

interface ArchInit {
  id: number;
  arch_id: string;
  name: string;
  source: string;
}

interface ArchEntry extends ArchInit {
  score: number;
  learnability: number;
  speed: number;
  fit_time: number;
  fight_count: number;
  color: string;
}

interface FightLogEntry {
  fight: number;
  i: number;
  j: number;
  failed: boolean;
  score_i: number;
  score_j: number;
  learn_i: number;
  learn_j: number;
  speed_i: number;
  speed_j: number;
  time_i: number;
  time_j: number;
  loss_i: number;
  loss_j: number;
}

interface UploadedArch {
  arch_id: string;
  filename: string;
}

type ChartMetric = 'score' | 'learnability' | 'speed';

const metricLabels: Record<ChartMetric, string> = {
  score:        'Combined',
  learnability: 'Learnability',
  speed:        'Speed',
};

const S: Record<string, React.CSSProperties> = {
  glassCard: {
    background: 'var(--panel-bg)',
    border: '1px solid var(--glass-border)',
    borderRadius: '8px',
    boxShadow: 'var(--window-shadow)',
    backdropFilter: 'blur(14px)',
    position: 'relative',
    overflow: 'hidden',
  },
  cardAccent: {
    position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
    background: 'linear-gradient(90deg, transparent, var(--theme-primary), var(--theme-accent), transparent)',
  },
  label: { fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 800, textTransform: 'uppercase' as const },
  input: {
    ...INPUT_STYLE,
    transition: 'border-color 0.15s, box-shadow 0.15s',
  } as React.CSSProperties,
};

const glassCardStyle: React.CSSProperties = {
  ...S.glassCard,
  padding: '24px',
};

//  component 

export default function TournamentViewer() {
  // pool config
  const [nRandom, setNRandom]               = useState(6);
  const [uploadedArchs, setUploadedArchs]   = useState<UploadedArch[]>([]);
  const [uploading, setUploading]           = useState(false);
  const [moduleSet, setModuleSet]           = useState('Unified');
  const [simpOppBal, setSimpOppBal]         = useState(0.2);
  const [maxDuration, setMaxDuration]       = useState(600);
  const [archSize, setArchSize]             = useState(12);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // tournament state
  const [status, setStatus]           = useState<'idle' | 'running' | 'done' | 'error'>('idle');
  const [progress, setProgress]       = useState({ current: 0, total: 0 });
  const [log, setLog]                 = useState('Configure your pool and start the tournament.');
  const [leaderboard, setLeaderboard] = useState<ArchEntry[]>([]);
  const [fightLog, setFightLog]       = useState<FightLogEntry[]>([]);
  const [poolSize, setPoolSize]       = useState(0);
  const [chartMetric, setChartMetric] = useState<ChartMetric>('score');
  const [expandedFight, setExpandedFight] = useState<number | null>(null);

  // refs
  const wsRef       = useRef<WebSocket | null>(null);
  const statusRef   = useRef(status);
  const archInfoRef = useRef<ArchInit[]>([]);
  const logEndRef   = useRef<HTMLDivElement | null>(null);

  useEffect(() => { statusRef.current = status; }, [status]);
  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [fightLog]);

  /*  upload .pkl files ─ */
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    const newArchs: UploadedArch[] = [];

    for (const file of Array.from(files)) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${API}/api/upload_arch`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        newArchs.push({
          arch_id: data.arch_id,
          filename: file.name.replace(/\.pkl$/i, ''),
        });
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err);
      }
    }

    setUploadedArchs(prev => [...prev, ...newArchs]);
    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeUploaded = (archId: string) => {
    setUploadedArchs(prev => prev.filter(a => a.arch_id !== archId));
  };

  const expectedPool   = nRandom + uploadedArchs.length;
  const expectedFights = (expectedPool * (expectedPool - 1)) / 2;

  // reset
  const resetToIdle = () => {
    setStatus('idle');
    setLeaderboard([]);
    setFightLog([]);
    setProgress({ current: 0, total: 0 });
    setPoolSize(0);
    setChartMetric('score');
    setExpandedFight(null);
    setLog('Configure your pool and start the tournament.');
  };

  /*  start tournament  */
  const startTournament = () => {
    if (expectedPool < 2) return;

    setStatus('running');
    setLog('Generating architectures…');
    setFightLog([]);
    setLeaderboard([]);
    setExpandedFight(null);
    setProgress({ current: 0, total: expectedPool });

    wsRef.current?.close();
    const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE_URL}/ws/tournament`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({
        n_random: nRandom,
        module_set: moduleSet,
        client_id: getClientId(),
        simp_opp_bal: simpOppBal,
        max_duration: maxDuration,
        arch_size: archSize,
        loaded_arch_ids: uploadedArchs.map(a => ({
          arch_id: a.arch_id,
          name: a.filename,
        })),
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'generation_progress') {
        setProgress({ current: data.current, total: data.total });
        setLog(`Generating architectures… ${data.current}/${data.total}`);
      }

      else if (data.type === 'init') {
        archInfoRef.current = data.architectures;
        setPoolSize(data.n_archs);
        setProgress({ current: 0, total: data.total_fights });
        if (data.normalization_group) {
          setLog(`Tournament ready · normalization: ${data.normalization_group}`);
        }
        setLeaderboard(
          data.architectures.map((a: ArchInit, i: number) => ({
            ...a,
            score: 0, learnability: 0, speed: 0, fit_time: 0, fight_count: 0,
            color: `hsl(${22 + ((i * 34) % 70)}, 95%, ${54 + (i % 3) * 8}%)`,
          })),
        );
        setLog('Tournament started — fights in progress…');
      }

      else if (data.type === 'fight_result') {
        setProgress({ current: data.fight, total: data.total });

        const info  = archInfoRef.current;
        const nameI = info[data.i]?.name ?? `#${data.i}`;
        const nameJ = info[data.j]?.name ?? `#${data.j}`;

        setLog(
          data.failed
            ? `Fight ${data.fight}/${data.total}: ${nameI} vs ${nameJ} fallback`
            : `Fight ${data.fight}/${data.total}: ${nameI} vs ${nameJ}`,
        );

        setFightLog(prev => [...prev, {
          fight: data.fight, i: data.i, j: data.j,
          failed: data.failed,
          score_i: data.score_i, score_j: data.score_j,
          learn_i: data.learn_i, learn_j: data.learn_j,
          speed_i: data.speed_i, speed_j: data.speed_j,
          time_i:  data.time_i,  time_j:  data.time_j,
          loss_i:  data.loss_i,  loss_j:  data.loss_j,
        }]);

        setLeaderboard(prev =>
          prev.map(e => ({
            ...e,
            score:        data.scores[e.id],
            learnability: data.learnabilities[e.id],
            speed:        data.speeds[e.id],
            fit_time:     data.fit_times[e.id],
            fight_count:  data.fight_counts[e.id],
          })).sort((a, b) => b.score - a.score),
        );
      }

      else if (data.type === 'done') {
        statusRef.current = 'done';
        setStatus('done');
        setLog('Tournament Complete!');
        ws.close();
      }
    };

    ws.onerror = () => { setStatus('error'); setLog('Connection error — is the backend running?'); };
    ws.onclose = () => {
      if (statusRef.current === 'running') { setStatus('error'); setLog('Connection closed unexpectedly.'); }
    };
  };

  useEffect(() => () => { wsRef.current?.close(); }, []);

  // derived
  const pct    = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;
  const winner = status === 'done' && leaderboard.length > 0 ? leaderboard[0] : null;
  const chartData = [...leaderboard].sort((a, b) => b[chartMetric] - a[chartMetric]);

  const progressColor = status === 'done' ? '#66ff99'
    : status === 'error' ? '#ff4d2e'
    : 'var(--theme-primary)';
  const progressGlow = status === 'done' ? 'rgba(102,255,153,0.28)'
    : status === 'error' ? 'rgba(255,77,46,0.3)'
    : 'rgba(var(--theme-primary-rgb),0.3)';

  //  render 
  return (
    <div className="page-content">

      {/* toolbar — minimal when idle */}
      <div className="page-toolbar">
        <h2 style={{ fontSize: '16px', fontWeight: 800, color: 'var(--theme-primary)', textShadow: '0 0 18px rgba(var(--theme-primary-rgb),0.35)', margin: 0 }}>
          Tournament Arena
        </h2>
        <div style={{ flex: 1 }} />
        {(status === 'done' || status === 'error') && (
          <button className="btn btn-back" onClick={resetToIdle}>New Tournament</button>
        )}
        {status === 'running' && (
          <span style={{ color: '#60a5fa', fontSize: '12px' }}>Running…</span>
        )}
      </div>

      <div className="graph-container" style={{ overflowY: 'auto' }}>

        {/*  idle — pool configuration ─ */}
        {status === 'idle' && (
          <div style={{ padding: '24px', maxWidth: '640px', margin: '0 auto' }}>
            <div style={{ ...glassCardStyle }}>
              <div style={S.cardAccent} />
              <h3 style={{ color: '#f1f5f9', margin: '0 0 20px 0', fontSize: '15px', fontWeight: 700 }}>
                Pool Configuration
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>

                {/* Random count */}
                <div>
                  <label style={S.label}>Random Architectures</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '6px' }}>
                    <LooseNumberInput
                      min={0} max={32} value={nRandom}
                      onChange={setNRandom}
                      fallback={0}
                      style={{ ...S.input, width: '80px' }}
                    />
                    <span style={{ color: '#64748b', fontSize: '12px' }}>Generate from scratch</span>
                  </div>
                </div>

                {/* Opponent simplicity */}
                <div>
                  <label style={S.label}>Opponent Simplicity</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '6px' }}>
                    <input
                      type="range" min={0} max={1} step={0.05} value={simpOppBal}
                      onChange={e => setSimpOppBal(parseFloat(e.target.value))}
                      style={{ flex: 1, accentColor: '#818cf8' }}
                    />
                    <span style={{
                      minWidth: '36px', textAlign: 'center', fontWeight: 700,
                      color: '#a5b4fc', fontSize: '14px',
                    }}>{simpOppBal.toFixed(2)}</span>
                  </div>
                  <p style={{ margin: '4px 0 0', color: '#475569', fontSize: '11px' }}>
                    Rewards architectures that best learn simple (easy-to-learn) opponents
                  </p>
                </div>

                {/* Architecture size */}
                <div>
                  <label style={S.label}>Architecture Nodes</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '6px' }}>
                    <input
                      type="range" min={7} max={15} step={1} value={archSize}
                      onChange={e => setArchSize(parseInt(e.target.value))}
                      style={{ flex: 1, accentColor: '#818cf8' }}
                    />
                    <span style={{
                      minWidth: '28px', textAlign: 'center', fontWeight: 700,
                      color: '#a5b4fc', fontSize: '14px',
                    }}>{archSize}</span>
                  </div>
                  <p style={{ margin: '4px 0 0', color: '#475569', fontSize: '11px' }}>
                    Number of nodes in each randomly generated architecture
                  </p>
                </div>

                {/* Upload */}
                <div>
                  <label style={S.label}>Inject Saved Architectures</label>
                  <div style={{ marginTop: '6px' }}>
                    <input
                      ref={fileInputRef}
                      type="file" accept=".pkl" multiple
                      onChange={handleFileUpload}
                      style={{ display: 'none' }}
                    />
                    <button
                      className="btn btn-back"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploading}
                      style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                      {uploading ? 'Uploading…' : 'Upload .pkl'}
                    </button>
                  </div>
                  {uploadedArchs.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '8px' }}>
                      {uploadedArchs.map(arch => (
                        <span key={arch.arch_id} style={{
                          display: 'inline-flex', alignItems: 'center', gap: '6px',
                          padding: '4px 10px', borderRadius: '10px',
                          background: 'rgba(51, 65, 85, 0.4)', border: '1px solid rgba(148,163,184,0.08)',
                          color: '#e2e8f0', fontSize: '12px',
                        }}
                          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(71,85,105,0.5)'; }}
                          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(51,65,85,0.4)'; }}
                        >
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                          </svg>
                          {arch.filename}.pkl
                          <button onClick={() => removeUploaded(arch.arch_id)}
                            style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '14px', padding: 0, lineHeight: 1 }}
                          >×</button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Max fight duration */}
                <div>
                  <label style={S.label}>Fight Timeout</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '6px' }}>
                    <LooseNumberInput
                      min={60} max={3600} step={30} value={maxDuration}
                      onChange={setMaxDuration}
                      fallback={600}
                      style={{ ...S.input, width: '80px' }}
                    />
                    <span style={{ color: '#64748b', fontSize: '12px' }}>seconds ({Math.round(maxDuration / 60)} min)</span>
                  </div>
                </div>

                {/* Module set + start row */}
                <div style={{
                  marginTop: '8px', padding: '16px', borderRadius: '12px',
                  background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(148,163,184,0.06)',
                  display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <label style={S.label}>Module Set</label>
                    <select value={moduleSet} onChange={e => setModuleSet(e.target.value)}
                      style={{
                        background: 'rgba(15,23,42,0.8)', color: '#f1f5f9',
                        border: '1px solid rgba(148,163,184,0.12)', borderRadius: '8px',
                        padding: '6px 12px', fontSize: '12px', outline: 'none', cursor: 'pointer',
                      }}>
                      {MODULE_SET_OPTIONS.map(opt => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>
                  <div style={{ flex: 1 }} />
                  <button
                    className="btn btn-primary"
                    onClick={startTournament}
                    disabled={expectedPool < 2}
                    style={{ fontSize: '14px', padding: '9px 24px' }}
                  >
                    Start Tournament
                  </button>
                  {expectedPool < 2 && (
                    <span style={{ color: '#f59e0b', fontSize: '11px' }}>Need ≥2 archs</span>
                  )}
                </div>

                {/* summary */}
                <div style={{
                  padding: '10px 16px', borderRadius: '10px',
                  background: 'rgba(15,23,42,0.4)',
                  display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px',
                  fontSize: '13px', color: 'var(--text-secondary)',
                }}>
                  <span>
                    Pool: <strong style={{ color: '#f1f5f9' }}>{expectedPool}</strong> architectures
                    {uploadedArchs.length > 0 && (
                      <span style={{ color: '#64748b', fontSize: '11px' }}>
                        {' '}({nRandom} random + {uploadedArchs.length} uploaded)
                      </span>
                    )}
                  </span>
                  <span>
                    Fights: <strong style={{ color: '#60a5fa' }}>{expectedFights}</strong>
                  </span>
                  <span style={{ color: '#475569', fontSize: '11px' }}>
                    {simpOppBal > 0 ? `simplicity ×${simpOppBal.toFixed(2)}` : 'no simplicity weighting'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/*  progress bar ─ */}
        {status !== 'idle' && (
          <div style={{ padding: '16px 24px', maxWidth: '800px', margin: '0 auto' }}>
            <div style={{ ...glassCardStyle, padding: '16px 20px' }}>
              <div style={S.cardAccent} />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{
                  color: status === 'error' ? 'var(--danger)' : 'var(--text-secondary)', fontSize: '13px',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{log}</span>
                <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: '13px', fontVariantNumeric: 'tabular-nums', flexShrink: 0, marginLeft: '12px' }}>
                  {pct}%
                </span>
              </div>
              <div style={{
                width: '100%', height: '8px', backgroundColor: 'rgba(15,23,42,0.6)',
                borderRadius: '6px', overflow: 'hidden', position: 'relative',
              }}>
                <div style={{
                  width: `${pct}%`, height: '100%', borderRadius: '6px',
                  transition: 'width 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                  background: `linear-gradient(90deg, ${progressColor}, ${progressColor}dd)`,
                  boxShadow: `0 0 12px ${progressGlow}`,
                  position: 'relative',
                }}>
                  {status === 'running' && (
                    <div style={{
                      position: 'absolute', inset: 0,
                      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent)',
                      animation: 'shimmer 1.5s ease-in-out infinite',
                    }} />
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/*  error — arch download panel  */}
        {status === 'error' && archInfoRef.current.length > 0 && (
          <div style={{ padding: '0 24px', maxWidth: '640px', margin: '0 auto' }}>
            <div style={{ ...glassCardStyle, padding: '20px' }}>
              <div style={S.cardAccent} />
              <h3 style={{ color: '#f87171', margin: '0 0 6px 0', fontSize: '15px', fontWeight: 700 }}>
                Tournament Error
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '12px', margin: '0 0 16px 0' }}>
                A fight exceeded the {Math.round(maxDuration / 60)}-minute timeout. Download the architectures to inspect what went wrong.
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {archInfoRef.current.map((a: ArchInit) => (
                  <button
                    key={a.arch_id}
                    onClick={() => downloadArch(a.arch_id, a.name.replace(/\s+/g, '_'))}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '6px',
                      padding: '8px 16px', background: 'rgba(239,68,68,0.08)',
                      border: '1px solid rgba(239,68,68,0.15)', borderRadius: '10px',
                      color: '#fca5a5', cursor: 'pointer', fontSize: '13px', fontWeight: 600,
                      transition: 'all 0.15s',
                    }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.15)'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.08)'; }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    {a.name}.pkl
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/*  winner banner  */}
        {winner && (() => {
          const fc = Math.max(winner.fight_count, 1);
          return (
            <div style={{ padding: '0 24px', maxWidth: '900px', margin: '0 auto' }}>
              <div style={{
                position: 'relative', overflow: 'hidden',
                background: 'linear-gradient(135deg, rgba(15,23,42,0.85), rgba(30,27,10,0.7))',
                border: '1px solid rgba(251,191,36,0.2)', borderRadius: '20px',
                padding: '24px 28px', textAlign: 'center',
                boxShadow: '0 8px 40px rgba(0,0,0,0.3), 0 0 60px rgba(251,191,36,0.06)',
              }}>
                <div style={{
                  position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
                  background: 'linear-gradient(90deg, transparent, rgba(251,191,36,0.5), transparent)',
                }} />
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '12px' }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5C7 4 7 7 9.5 7" />
                    <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5C17 4 17 7 14.5 7" />
                    <path d="M4 22h16" />
                    <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22" />
                    <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22" />
                    <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z" />
                  </svg>
                  <h3 style={{ color: '#fbbf24', margin: 0, fontSize: '18px', fontWeight: 800 }}>
                    Winner: {winner.name}
                    {winner.source === 'uploaded' && <span style={{ fontSize: '12px', color: '#fde68a', fontWeight: 400 }}> (uploaded)</span>}
                  </h3>
                  <button
                    onClick={() => downloadArch(winner.arch_id, winner.name.replace(/\s+/g, '_'))}
                    title="Save .pkl"
                    style={{
                      display: 'flex', alignItems: 'center', gap: '4px',
                      padding: '5px 12px', background: 'rgba(251,191,36,0.12)',
                      border: '1px solid rgba(251,191,36,0.2)', borderRadius: '8px',
                      color: '#fbbf24', cursor: 'pointer', fontSize: '11px', fontWeight: 600,
                      transition: 'all 0.15s',
                    }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(251,191,36,0.2)'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(251,191,36,0.12)'; }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    .pkl
                  </button>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', marginTop: '10px', fontSize: '12px', color: '#d4a574', flexWrap: 'wrap' }}>
                  <span>Score: <strong style={{ color: '#fde68a' }}>{winner.score.toFixed(3)}</strong></span>
                  <span>Learn: <strong style={{ color: '#fde68a' }}>{winner.learnability.toFixed(3)}</strong></span>
                  <span>Speed: <strong style={{ color: '#fde68a' }}>{winner.speed.toFixed(3)}</strong></span>
                  <span>Avg Time: <strong style={{ color: '#fde68a' }}>{(winner.fit_time / fc).toFixed(2)}s</strong></span>
                </div>
              </div>
            </div>
          );
        })()}

        {/*  leaderboard + fight log  */}
        {leaderboard.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: '16px', padding: '16px 24px' }}>

            {/* leaderboard chart */}
            <div style={{ ...S.glassCard, padding: '16px', display: 'flex', flexDirection: 'column', minHeight: '300px' }}>
              <div style={S.cardAccent} />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexShrink: 0 }}>
                <h3 style={{ color: '#f1f5f9', margin: 0, fontSize: '14px', fontWeight: 700 }}>Leaderboard</h3>
                <div style={{ display: 'flex', gap: '4px' }}>
                  {(Object.keys(metricLabels) as ChartMetric[]).map(m => (
                    <button
                      key={m}
                      onClick={() => setChartMetric(m)}
                      style={{
                        padding: '3px 10px', borderRadius: '6px', fontSize: '11px', border: 'none', cursor: 'pointer',
                        background: chartMetric === m
                          ? 'linear-gradient(135deg, rgba(var(--theme-primary-rgb),0.32), rgba(var(--theme-accent-rgb),0.14))'
                          : 'rgba(0,0,0,0.34)',
                        color: chartMetric === m ? 'var(--theme-accent)' : 'var(--text-muted)',
                        fontWeight: chartMetric === m ? 600 : 400,
                        transition: 'all 0.15s',
                      }}
                    >{metricLabels[m]}</button>
                  ))}
                </div>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 50, left: 10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" horizontal={false} />
                    <XAxis type="number" stroke="#475569" tick={{ fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(1)} />
                    <YAxis type="category" dataKey="name" stroke="#475569" width={90} tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.12)',
                        borderRadius: '8px', fontSize: '12px',
                      }}
                      itemStyle={{ color: 'var(--text-secondary)' }}
                      labelStyle={{ color: '#f1f5f9', fontWeight: 700 }}
                      formatter={(value: any) => [typeof value === 'number' ? value.toFixed(3) : value, metricLabels[chartMetric]]}
                    />
                    <Bar dataKey={chartMetric} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                      {chartData.map(e => <Cell key={e.id} fill={e.color} />)}
                      <LabelList
                        dataKey={chartMetric} position="right"
                        style={{ fill: '#e2e8f0', fontSize: '10px', fontWeight: 600 }}
                        formatter={(v: any) => (typeof v === 'number' ? v.toFixed(2) : v)}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* fight log */}
            <div style={{ ...S.glassCard, padding: '16px', display: 'flex', flexDirection: 'column', minHeight: '300px' }}>
              <div style={S.cardAccent} />
              <h3 style={{ color: '#f1f5f9', margin: '0 0 12px 0', fontSize: '14px', fontWeight: 700, flexShrink: 0 }}>Fight Log</h3>
              <div style={{
                flex: 1, overflowY: 'auto', minHeight: 0,
                fontSize: '12px', fontFamily: 'monospace', display: 'flex', flexDirection: 'column', gap: '1px',
              }}>
                {fightLog.length === 0 && <span style={{ color: '#475569', fontSize: '12px' }}>Fights will appear here…</span>}

                {fightLog.map(f => {
                  const info  = archInfoRef.current;
                  const nameI = info[f.i]?.name ?? `#${f.i}`;
                  const nameJ = info[f.j]?.name ?? `#${f.j}`;
                  const iWins = f.score_i > f.score_j;
                  const isLatest  = f.fight === fightLog[fightLog.length - 1]?.fight;
                  const isExpanded = expandedFight === f.fight;

                  return (
                    <div key={f.fight}>
                      <div
                        onClick={() => setExpandedFight(isExpanded ? null : f.fight)}
                        style={{
                          display: 'flex', gap: '8px', padding: '6px 10px', borderRadius: '6px', cursor: 'pointer',
                          background: isLatest ? 'rgba(var(--theme-primary-rgb),0.12)' : isExpanded ? 'rgba(var(--theme-primary-rgb),0.08)' : 'transparent',
                          border: isLatest ? '1px solid rgba(var(--theme-primary-rgb),0.2)' : '1px solid transparent',
                          transition: 'background 0.15s',
                        }}
                        onMouseEnter={e => { if (!isLatest && !isExpanded) (e.currentTarget as HTMLElement).style.background = 'rgba(var(--theme-primary-rgb),0.1)'; }}
                        onMouseLeave={e => { if (!isLatest && !isExpanded) (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
                      >
                        <span style={{ color: '#475569', minWidth: '32px' }}>#{f.fight}</span>
                        {f.failed && <span style={{ color: '#f59e0b' }} title="Used fallback scores">!</span>}
                        <span style={{ color: iWins ? 'var(--success)' : 'var(--text-secondary)', fontWeight: iWins ? 700 : 400 }}>
                          {nameI} ({f.score_i.toFixed(2)})
                        </span>
                        <span style={{ color: '#475569' }}>vs</span>
                        <span style={{ color: !iWins ? 'var(--success)' : 'var(--text-secondary)', fontWeight: !iWins ? 700 : 400 }}>
                          {nameJ} ({f.score_j.toFixed(2)})
                        </span>
                        <span style={{ color: '#475569', marginLeft: 'auto', fontSize: '10px' }}>
                          {isExpanded ? '▲' : '▼'}
                        </span>
                      </div>

                      {isExpanded && (
                        <div style={{
                          display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px',
                          padding: '8px 8px 10px 44px', fontSize: '11px', color: 'var(--text-secondary)',
                        }}>
                          {[
                            { name: nameI, score: f.score_i, learn: f.learn_i, speed: f.speed_i, time: f.time_i, loss: f.loss_i },
                            { name: nameJ, score: f.score_j, learn: f.learn_j, speed: f.speed_j, time: f.time_j, loss: f.loss_j },
                          ].map((side, idx) => (
                            <div key={idx} style={{
                              background: 'rgba(15,23,42,0.5)', padding: '8px 12px', borderRadius: '8px',
                              border: '1px solid rgba(148,163,184,0.06)',
                            }}>
                              <div style={{ color: '#f1f5f9', fontWeight: 600, marginBottom: '6px', fontSize: '12px' }}>{side.name}</div>
                              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 16px' }}>
                                <span>Score</span><strong style={{ color: '#e2e8f0', textAlign: 'right' }}>{side.score.toFixed(3)}</strong>
                                <span>Learn</span><strong style={{ color: '#38bdf8', textAlign: 'right' }}>{side.learn.toFixed(3)}</strong>
                                <span>Speed</span><strong style={{ color: 'var(--theme-primary)', textAlign: 'right' }}>{side.speed.toFixed(3)}</strong>
                                <span>Time</span><strong style={{ color: '#e2e8f0', textAlign: 'right' }}>{side.time.toFixed(2)}s</strong>
                                <span>Loss</span><strong style={{ color: '#e2e8f0', textAlign: 'right' }}>{side.loss < 100 ? side.loss.toFixed(5) : side.loss.toExponential(2)}</strong>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
                <div ref={logEndRef} />
              </div>
            </div>
          </div>
        )}

        {/*  final standings table  */}
        {status === 'done' && leaderboard.length > 0 && (
          <div style={{ padding: '0 24px 24px', maxWidth: '900px', margin: '0 auto' }}>
            <div style={{ ...glassCardStyle, padding: '20px' }}>
              <div style={S.cardAccent} />
              <h3 style={{ color: '#f1f5f9', margin: '0 0 16px 0', fontSize: '15px', fontWeight: 700 }}>Final Standings</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', minWidth: '700px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(148,163,184,0.08)', color: '#64748b' }}>
                      <th style={{ textAlign: 'left',  padding: '8px 10px' }}>Rank</th>
                      <th style={{ textAlign: 'left',  padding: '8px 10px' }}>Architecture</th>
                      <th style={{ textAlign: 'left',  padding: '8px 10px' }}>Source</th>
                      <th style={{ textAlign: 'right', padding: '8px 10px' }}>Score</th>
                      <th style={{ textAlign: 'right', padding: '8px 10px' }}>Learn</th>
                      <th style={{ textAlign: 'right', padding: '8px 10px' }}>Speed</th>
                      <th style={{ textAlign: 'right', padding: '8px 10px' }}>Avg Time</th>
                      <th style={{ textAlign: 'right', padding: '8px 10px' }}>Fights</th>
                      <th style={{ textAlign: 'right', padding: '8px 10px' }} />
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.map((entry, rank) => {
                      const fc = Math.max(entry.fight_count, 1);
                      return (
                        <tr key={entry.id} style={{
                          borderBottom: '1px solid rgba(148,163,184,0.04)',
                          background: rank === 0 ? 'rgba(251,191,36,0.04)' : 'transparent',
                          transition: 'background 0.15s',
                        }}
                          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(var(--theme-primary-rgb),0.08)'; }}
                          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = rank === 0 ? 'rgba(251,191,36,0.04)' : 'transparent'; }}
                        >
                          <td style={{ padding: '8px 10px', color: rank === 0 ? '#fbbf24' : '#64748b', fontWeight: rank === 0 ? 700 : 400 }}>
                            {rank === 0 ? '🥇' : rank === 1 ? '🥈' : rank === 2 ? '🥉' : `#${rank + 1}`}
                          </td>
                          <td style={{ padding: '8px 10px', color: '#f1f5f9' }}>
                            <span style={{
                              display: 'inline-block', width: '8px', height: '8px',
                              borderRadius: '50%', backgroundColor: entry.color, marginRight: '8px',
                            }} />
                            {entry.name}
                          </td>
                          <td style={{ padding: '8px 10px', color: '#64748b', fontSize: '12px' }}>
                            <span style={{
                              padding: '2px 8px', borderRadius: '6px', fontSize: '10px', fontWeight: 600,
                              background: entry.source === 'uploaded' ? 'rgba(var(--theme-accent-rgb),0.12)' : 'rgba(var(--theme-primary-rgb),0.12)',
                              color: entry.source === 'uploaded' ? 'var(--theme-accent)' : 'var(--theme-primary)',
                            }}>
                              {entry.source === 'uploaded' ? 'Uploaded' : 'Random'}
                            </span>
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: '#f1f5f9', fontWeight: 600 }}>
                            {entry.score.toFixed(3)}
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: '#38bdf8', fontWeight: 600 }}>
                            {entry.learnability.toFixed(3)}
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: 'var(--theme-primary)', fontWeight: 600 }}>
                            {entry.speed.toFixed(3)}
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: '#64748b' }}>
                            {(entry.fit_time / fc).toFixed(2)}s
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: '#475569' }}>
                            {entry.fight_count}/{poolSize - 1}
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                            <button
                              onClick={() => downloadArch(entry.arch_id, entry.name.replace(/\s+/g, '_'))}
                              title="Download .pkl"
                              style={{
                                display: 'flex', alignItems: 'center', gap: '3px',
                                padding: '4px 10px', background: 'rgba(51,65,85,0.3)',
                                border: '1px solid rgba(148,163,184,0.08)', borderRadius: '6px',
                                color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '11px',
                                transition: 'all 0.15s',
                              }}
                              onMouseEnter={e => { const el = e.currentTarget as HTMLElement; el.style.background = 'rgba(71,85,105,0.4)'; el.style.color = '#e2e8f0'; }}
                              onMouseLeave={e => { const el = e.currentTarget as HTMLElement; el.style.background = 'rgba(0,0,0,0.34)'; el.style.color = 'var(--text-secondary)'; }}
                            >
                              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                              </svg>
                              .pkl
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
