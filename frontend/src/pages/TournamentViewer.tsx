import { useState, useEffect, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts';
import SaveArchButton from '../components/SaveArchButton';

// types

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

type ChartMetric = 'score' | 'learnability' | 'speed';

const metricLabels: Record<ChartMetric, string> = {
  score:        'Combined',
  learnability: 'Learnability',
  speed:        'Speed',
};

// component

export default function TournamentViewer() {
  // pool config
  const [nRandom, setNRandom]             = useState(6);
  const [pklFiles, setPklFiles]           = useState<string[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [dropdownValue, setDropdownValue] = useState('');

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

  // fetch saved files
  const fetchFiles = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/saved_archs');
      if (res.ok) { const d = await res.json(); setPklFiles(d.files || []); }
    } catch (e) { console.error('Could not load pkl files', e); }
  };
  useEffect(() => { fetchFiles(); }, []);

  // pool helpers
  const addFile = () => {
    if (dropdownValue && !selectedFiles.includes(dropdownValue)) {
      setSelectedFiles(prev => [...prev, dropdownValue]);
      setDropdownValue('');
    }
  };
  const removeFile = (f: string) => setSelectedFiles(prev => prev.filter(x => x !== f));

  const expectedPool   = nRandom + selectedFiles.length;
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

  // start tournament
  const startTournament = () => {
    if (expectedPool < 2) return;

    setStatus('running');
    setLog('Generating architectures (takes ~10 seconds per architecture)…');
    setFightLog([]);
    setLeaderboard([]);
    setExpandedFight(null);
    setProgress({ current: 0, total: expectedFights });

    wsRef.current?.close();
    const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE_URL}/ws/tournament`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ n_random: nRandom, loaded_archs: selectedFiles }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'init') {
        archInfoRef.current = data.architectures;
        setPoolSize(data.n_archs);
        setProgress({ current: 0, total: data.total_fights });
        setLeaderboard(
          data.architectures.map((a: ArchInit, i: number) => ({
            ...a,
            score: 0, learnability: 0, speed: 0, fit_time: 0, fight_count: 0,
            color: `hsl(${(i * 360) / data.n_archs}, 70%, 60%)`,
          })),
        );
        setLog('Tournament started — fights in progress (takes ~50 seconds per fight)…');
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
            score:         data.scores[e.id],
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

  // chart data sorted by selected metric
  const chartData = [...leaderboard].sort((a, b) => b[chartMetric] - a[chartMetric]);

  // render
  return (
    <div className="page-content">

      {/* toolbar */}
      <div className="page-toolbar" style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', alignItems: 'center' }}>
        <h2 style={{ fontSize: '18px', color: '#f1f5f9', margin: 0 }}>Tournament Arena</h2>
        <div style={{ width: '1px', height: '24px', background: '#334155' }} />
        <button
          className="btn btn-primary"
          onClick={startTournament}
          disabled={status === 'running' || expectedPool < 2}
        >
          {status === 'running'
            ? 'Running…'
            : `Start (${expectedPool} archs -> ${expectedFights} fights)`}
        </button>
        {(status === 'done' || status === 'error') && (
          <button className="btn btn-back" onClick={resetToIdle}>New Tournament</button>
        )}
        {expectedPool < 2 && status === 'idle' && (
          <span style={{ color: '#f59e0b', fontSize: '12px' }}>Need at least 2 architectures</span>
        )}
      </div>

      <div className="graph-container" style={{ padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>

        {/* pool configuration */}
        {status === 'idle' && (
          <div style={{ backgroundColor: '#1e293b', padding: '20px', borderRadius: '8px' }}>
            <h3 style={{ color: 'white', margin: '0 0 16px 0', fontSize: '15px' }}>Pool Configuration</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <label style={{ color: '#94a3b8', fontSize: '13px', minWidth: '160px' }}>Random Architectures:</label>
                <input
                  type="number" min={0} max={32} value={nRandom}
                  onChange={e => setNRandom(Math.max(0, parseInt(e.target.value) || 0))}
                  style={{
                    width: '70px', padding: '6px 10px', borderRadius: '4px',
                    background: '#0f172a', color: 'white', border: '1px solid #475569', fontSize: '13px',
                  }}
                />
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <label style={{ color: '#94a3b8', fontSize: '13px', minWidth: '160px' }}>Inject Saved Archs:</label>
                  <select
                    value={dropdownValue}
                    onChange={e => setDropdownValue(e.target.value)}
                    style={{
                      flex: 1, maxWidth: '250px', padding: '6px', borderRadius: '4px',
                      background: '#0f172a', color: 'white', border: '1px solid #475569', fontSize: '13px',
                    }}
                  >
                    <option value="">Select a .pkl file…</option>
                    {pklFiles.filter(f => !selectedFiles.includes(f)).map(f => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>
                  <button onClick={addFile} disabled={!dropdownValue}
                    style={{
                      padding: '6px 12px', borderRadius: '4px', fontSize: '13px',
                      background: dropdownValue ? '#3b82f6' : '#334155',
                      color: 'white', border: 'none', cursor: dropdownValue ? 'pointer' : 'default',
                    }}
                  >+ Add</button>
                  <button onClick={fetchFiles} title="Refresh list"
                    style={{
                      background: '#334155', border: '1px solid #475569', borderRadius: '4px',
                      padding: '5px 8px', cursor: 'pointer', color: 'white', fontSize: '13px',
                    }}
                  >Refresh</button>
                </div>
                {selectedFiles.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginLeft: '172px' }}>
                    {selectedFiles.map(file => (
                      <span key={file} style={{
                        display: 'inline-flex', alignItems: 'center', gap: '6px',
                        padding: '4px 10px', borderRadius: '12px',
                        backgroundColor: '#334155', color: '#e2e8f0', fontSize: '12px',
                      }}>
                        {file}
                        <button onClick={() => removeFile(file)}
                          style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '14px', padding: 0, lineHeight: 1 }}
                        >×</button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div style={{
                padding: '10px 16px', borderRadius: '6px', backgroundColor: '#0f172a',
                color: '#94a3b8', fontSize: '13px',
              }}>
                Total pool: <strong style={{ color: 'white' }}>{expectedPool}</strong> architectures
                → <strong style={{ color: 'white' }}>{expectedFights}</strong> fights
                {selectedFiles.length > 0 && <span> ({nRandom} random + {selectedFiles.length} saved)</span>}
                <br />
                <span style={{ fontSize: '11px', color: '#475569' }}>
                  Uses arena.get_scores() with random 3D tensors. Scores = learnability + speed, normalized against global baselines.
                </span>
              </div>
            </div>
          </div>
        )}

        {/* progress bar */}
        {status !== 'idle' && (
          <div style={{ backgroundColor: '#1e293b', padding: '16px 20px', borderRadius: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '13px' }}>
              <span style={{ color: status === 'error' ? '#ef4444' : '#94a3b8' }}>{log}</span>
              <span style={{ color: 'white', fontWeight: 700 }}>
                {progress.current}/{progress.total} ({pct}%)
              </span>
            </div>
            <div style={{ width: '100%', height: '8px', backgroundColor: '#0f172a', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                width: `${pct}%`, height: '100%', borderRadius: '4px', transition: 'width 0.3s ease',
                backgroundColor: status === 'done' ? '#10b981' : status === 'error' ? '#ef4444' : '#3b82f6',
              }} />
            </div>
          </div>
        )}

        {/* winner banner */}
        {winner && (() => {
          const fc = Math.max(winner.fight_count, 1);
          return (
            <div style={{
              background: 'linear-gradient(135deg, #78350f 0%, #92400e 50%, #78350f 100%)',
              border: '2px solid #fbbf24', borderRadius: '8px', padding: '20px', textAlign: 'center',
            }}>
              <h3 style={{ color: '#fbbf24', margin: '4px 0', fontSize: '20px' }}>
                Winner: {winner.name}
                {winner.source === 'loaded' && <span style={{ fontSize: '14px', color: '#fde68a' }}> (saved)</span>}
              </h3>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', marginTop: '8px', fontSize: '13px', color: '#fde68a' }}>
                <span>Score: <strong>{winner.score.toFixed(3)}</strong></span>
                <span>Learn: <strong>{winner.learnability.toFixed(3)}</strong></span>
                <span>Speed: <strong>{winner.speed.toFixed(3)}</strong></span>
                <span>Avg Time: <strong>{(winner.fit_time / fc).toFixed(2)}s</strong></span>
              </div>
            </div>
          );
        })()}

        {/* leaderboard + fight log */}
        {leaderboard.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>

            {/* leaderboard chart */}
            <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h3 style={{ color: 'white', margin: 0, fontSize: '15px' }}>Live Leaderboard</h3>
                <div style={{ display: 'flex', gap: '4px' }}>
                  {(Object.keys(metricLabels) as ChartMetric[]).map(m => (
                    <button
                      key={m}
                      onClick={() => setChartMetric(m)}
                      style={{
                        padding: '3px 10px', borderRadius: '4px', fontSize: '11px', border: 'none', cursor: 'pointer',
                        background: chartMetric === m ? '#3b82f6' : '#334155',
                        color: chartMetric === m ? 'white' : '#94a3b8',
                      }}
                    >{metricLabels[m]}</button>
                  ))}
                </div>
              </div>
              <div style={{ height: Math.max(280, leaderboard.length * 38) }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 50, left: 10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" stroke="#94a3b8" tickFormatter={(v: number) => v.toFixed(1)} />
                    <YAxis type="category" dataKey="name" stroke="#94a3b8" width={100} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '4px' }}
                      itemStyle={{ color: '#94a3b8', fontSize: '12px' }}
                      labelStyle={{ color: '#fff', marginBottom: '4px', fontWeight: 'bold' }}
                      formatter={(value: any) => [typeof value === 'number' ? value.toFixed(3) : value, metricLabels[chartMetric]]}
                    />
                    <Bar dataKey={chartMetric} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                      {chartData.map(e => <Cell key={e.id} fill={e.color} />)}
                      <LabelList
                        dataKey={chartMetric} position="right"
                        style={{ fill: '#e2e8f0', fontSize: '11px', fontWeight: 600 }}
                        formatter={(v: any) => (typeof v === 'number' ? v.toFixed(2) : v)}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* fight log */}
            <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px', display: 'flex', flexDirection: 'column' }}>
              <h3 style={{ color: 'white', margin: '0 0 12px 0', fontSize: '15px' }}>Fight Log</h3>
              <div style={{
                flex: 1, overflowY: 'auto', maxHeight: Math.max(250, leaderboard.length * 38 - 30),
                fontSize: '12px', fontFamily: 'monospace', display: 'flex', flexDirection: 'column', gap: '1px',
              }}>
                {fightLog.length === 0 && <span style={{ color: '#475569' }}>Fights will appear here…</span>}

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
                          display: 'flex', gap: '6px', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer',
                          backgroundColor: isLatest ? '#334155' : isExpanded ? '#2d3a4d' : 'transparent',
                        }}
                      >
                        <span style={{ color: '#64748b', minWidth: '36px' }}>#{f.fight}</span>
                        {f.failed && <span title="Used fallback scores">!</span>}
                        <span style={{ color: iWins ? '#10b981' : '#94a3b8', fontWeight: iWins ? 700 : 400 }}>
                          {nameI} ({f.score_i.toFixed(2)})
                        </span>
                        <span style={{ color: '#475569' }}>vs</span>
                        <span style={{ color: !iWins ? '#10b981' : '#94a3b8', fontWeight: !iWins ? 700 : 400 }}>
                          {nameJ} ({f.score_j.toFixed(2)})
                        </span>
                        <span style={{ color: '#475569', marginLeft: 'auto', fontSize: '10px' }}>
                          {isExpanded ? '^' : 'v'}
                        </span>
                      </div>

                      {isExpanded && (
                        <div style={{
                          display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px',
                          padding: '6px 8px 10px 44px', fontSize: '11px', color: '#94a3b8',
                        }}>
                          {[
                            { name: nameI, score: f.score_i, learn: f.learn_i, speed: f.speed_i, time: f.time_i, loss: f.loss_i },
                            { name: nameJ, score: f.score_j, learn: f.learn_j, speed: f.speed_j, time: f.time_j, loss: f.loss_j },
                          ].map((side, idx) => (
                            <div key={idx} style={{ backgroundColor: '#0f172a', padding: '6px 10px', borderRadius: '4px' }}>
                              <div style={{ color: 'white', fontWeight: 600, marginBottom: '4px' }}>{side.name}</div>
                              <div>Score: <strong style={{ color: '#e2e8f0' }}>{side.score.toFixed(3)}</strong></div>
                              <div>Learn: <strong style={{ color: '#38bdf8' }}>{side.learn.toFixed(3)}</strong></div>
                              <div>Speed: <strong style={{ color: '#a78bfa' }}>{side.speed.toFixed(3)}</strong></div>
                              <div>Time:  <strong style={{ color: '#e2e8f0' }}>{side.time.toFixed(2)}s</strong></div>
                              <div>Loss:  <strong style={{ color: '#e2e8f0' }}>{side.loss < 100 ? side.loss.toFixed(5) : side.loss.toExponential(2)}</strong></div>
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

        {/* final standings table */}
        {status === 'done' && leaderboard.length > 0 && (
          <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px' }}>
            <h3 style={{ color: 'white', margin: '0 0 12px 0', fontSize: '15px' }}>Final Standings</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', minWidth: '750px' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #334155', color: '#94a3b8' }}>
                    <th style={{ textAlign: 'left',  padding: '8px' }}>Rank</th>
                    <th style={{ textAlign: 'left',  padding: '8px' }}>Architecture</th>
                    <th style={{ textAlign: 'left',  padding: '8px' }}>Source</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Score</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Learn</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Speed</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Avg Time</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Fights</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Save</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((entry, rank) => {
                    const fc = Math.max(entry.fight_count, 1);
                    return (
                      <tr key={entry.id} style={{
                        borderBottom: '1px solid #1e293b',
                        backgroundColor: rank === 0 ? 'rgba(251,191,36,0.08)' : 'transparent',
                      }}>
                        <td style={{ padding: '8px', color: '#94a3b8' }}>
                          {'#' + (rank + 1)}
                        </td>
                        <td style={{ padding: '8px', color: 'white' }}>
                          <span style={{
                            display: 'inline-block', width: '10px', height: '10px',
                            borderRadius: '50%', backgroundColor: entry.color, marginRight: '8px',
                          }} />
                          {entry.name}
                        </td>
                        <td style={{ padding: '8px', color: '#94a3b8', fontSize: '12px' }}>
                          {entry.source === 'loaded' ? 'Saved' : 'Random'}
                        </td>
                        <td style={{ padding: '8px', textAlign: 'right', color: 'white', fontWeight: 600 }}>
                          {entry.score.toFixed(3)}
                        </td>
                        <td style={{ padding: '8px', textAlign: 'right', color: '#38bdf8', fontWeight: 600 }}>
                          {entry.learnability.toFixed(3)}
                        </td>
                        <td style={{ padding: '8px', textAlign: 'right', color: '#a78bfa', fontWeight: 600 }}>
                          {entry.speed.toFixed(3)}
                        </td>
                        <td style={{ padding: '8px', textAlign: 'right', color: '#94a3b8' }}>
                          {(entry.fit_time / fc).toFixed(2)}s
                        </td>
                        <td style={{ padding: '8px', textAlign: 'right', color: '#64748b' }}>
                          {entry.fight_count}/{poolSize - 1}
                        </td>
                        <td style={{ padding: '8px', textAlign: 'right' }}>
                          <SaveArchButton
                            archId={entry.arch_id}
                            defaultName={entry.name.replace(/\s+/g, '_')}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* idle placeholder */}
        {status === 'idle' && (
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            height: '180px', color: '#475569', gap: '8px',
          }}>
            <p style={{ fontSize: '14px', margin: 0 }}>Configure your pool above, then hit <strong>Start</strong>.</p>
          </div>
        )}
      </div>
    </div>
  );
}