import { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import SaveArchButton from '../components/SaveArchButton';

interface LeaderboardEntry {
  id: number;
  arch_id: string;
  label: string;
  score: number;
  color: string;
}

export default function TournamentViewer() {
  const [nRandom, setNRandom] = useState<number>(8);
  const [pklFiles, setPklFiles] = useState<string[]>([]);
  const [selectedPkls, setSelectedPkls] = useState<string[]>([]);
  
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [log, setLog] = useState<string>("Waiting to start...");
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch available .pkl files on load
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/saved_archs")
      .then(res => res.json())
      .then(data => setPklFiles(data.files || []))
      .catch(err => console.error("Could not load pkl files", err));
  }, []);

  const togglePkl = (file: string) => {
    if (selectedPkls.includes(file)) {
      setSelectedPkls(selectedPkls.filter(f => f !== file));
    } else {
      setSelectedPkls([...selectedPkls, file]);
    }
  };

  const startTournament = () => {
    setStatus("running");
    setLog("Generating architectures and preparing pool...");
    setProgress({ current: 0, total: 0 });
    setLeaderboard([]); // Clear until backend sends 'init'
    
    if (wsRef.current) wsRef.current.close();
    
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/tournament");
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ n_random: nRandom, saved_archs: selectedPkls }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "init") {
        // Backend responds with the generated pool (labels & UUIDs)
        const initialBoard = data.archs.map((arch: any, i: number) => ({
          id: arch.id,
          arch_id: arch.arch_id,
          label: arch.label,
          score: 0,
          color: `hsl(${(i * 360) / data.archs.length}, 70%, 60%)`
        }));
        setLeaderboard(initialBoard);
        setLog("Tournament Pool Ready. Fighting...");
      } 
      else if (data.type === "fight_result") {
        setProgress({ current: data.fight, total: data.total });
        
        setLeaderboard(prev => {
          if (prev.length === 0) return prev;
          const A = prev.find(p => p.id === data.i)?.label || data.i;
          const B = prev.find(p => p.id === data.j)?.label || data.j;
          setLog(`Round ${data.fight}: ${A} vs ${B}`);

          const newBoard = prev.map(entry => ({
            ...entry,
            score: data.leaderboard[entry.id]
          }));
          return newBoard.sort((a, b) => b.score - a.score);
        });
      } 
      else if (data.type === "done") {
        setStatus("done");
        setLog("Tournament Complete! 🏆");
        ws.close();
      }
    };

    ws.onerror = () => {
      setStatus("error");
      setLog("Connection error. Is the backend running?");
    };

    ws.onclose = () => {
      if (status === "running") {
        setStatus("error");
        setLog("Connection closed unexpectedly.");
      }
    };
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div className="page-content">
      {/* TOOLBAR */}
      <div className="page-toolbar" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', alignItems: 'center' }}>
          <h2 style={{ fontSize: '18px', color: '#f1f5f9', margin: 0 }}>Round-Robin Tournament</h2>
          <div style={{ width: '1px', height: '24px', background: '#334155' }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>Random Pool Size:</span>
            <input 
              type="number" 
              min="0" 
              max="32" 
              value={nRandom} 
              onChange={(e) => setNRandom(parseInt(e.target.value) || 0)}
              disabled={status === "running"}
              style={{ width: '60px', padding: '6px', borderRadius: '4px', background: '#0f172a', color: 'white', border: '1px solid #475569', fontSize: '13px' }}
            />
          </div>

          <button className="btn btn-primary" onClick={startTournament} disabled={status === "running" || (nRandom + selectedPkls.length < 2)}>
            {status === "running" ? '⚔️ Tournament Running...' : '🏆 Start Tournament'}
          </button>
        </div>

        {/* SAVED ARCHITECTURE INJECTOR */}
        {pklFiles.length > 0 && (
          <div style={{ backgroundColor: '#1e293b', padding: '10px 15px', borderRadius: '6px', border: '1px solid #334155' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8', display: 'block', marginBottom: '8px' }}>
              Inject Saved Architectures ({selectedPkls.length} selected):
            </span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
              {pklFiles.map(file => (
                <label key={file} style={{ fontSize: '13px', color: 'white', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={selectedPkls.includes(file)} 
                    onChange={() => togglePkl(file)} 
                    disabled={status === "running"}
                  />
                  💾 {file}
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="graph-container" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* PROGRESS AREA */}
        <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: status === 'error' ? '#ef4444' : '#e2e8f0', fontWeight: 'bold' }}>{log}</span>
            <span style={{ color: '#94a3b8', fontSize: '14px' }}>{progress.current} / {progress.total} Fights</span>
          </div>
          
          <div style={{ width: '100%', height: '10px', backgroundColor: '#0f172a', borderRadius: '5px', overflow: 'hidden' }}>
            <div 
              style={{ 
                height: '100%', backgroundColor: status === 'done' ? '#10b981' : '#3b82f6', 
                width: progress.total > 0 ? `${(progress.current / progress.total) * 100}%` : '0%',
                transition: 'width 0.3s ease'
              }} 
            />
          </div>
        </div>

        {/* CHART AREA */}
        {leaderboard.length > 0 && (
          <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px', height: '350px' }}>
            <h3 style={{ margin: '0 0 16px 0', color: 'white', fontSize: '16px' }}>Live Leaderboard (Combined Final Score)</h3>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={leaderboard} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="label" stroke="#94a3b8" interval={0} angle={-30} textAnchor="end" height={60} />
                <YAxis stroke="#94a3b8" tickFormatter={(v) => v.toFixed(1)} />
                <Tooltip 
                  cursor={{ fill: '#334155', opacity: 0.4 }}
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #475569', color: '#f1f5f9' }}
                  formatter={(value: any) => [typeof value === 'number' ? value.toFixed(3) : value, 'Score']}
                />
                <Bar dataKey="score" isAnimationActive={false} radius={[4, 4, 0, 0]}>
                  {leaderboard.map((entry) => (
                    <Cell key={entry.id} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* DETAILED LEADERBOARD & SAVE BUTTONS */}
        {leaderboard.length > 0 && (
          <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 16px 0', color: 'white', fontSize: '16px' }}>Rankings</h3>
            <table style={{ width: '100%', color: 'white', borderCollapse: 'collapse', fontSize: '14px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155', textAlign: 'left', color: '#94a3b8' }}>
                  <th style={{ padding: '8px' }}>Rank</th>
                  <th style={{ padding: '8px' }}>Architecture</th>
                  <th style={{ padding: '8px' }}>Combined Score</th>
                  <th style={{ padding: '8px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((entry, index) => (
                  <tr key={entry.id} style={{ borderBottom: '1px solid #334155' }}>
                    <td style={{ padding: '8px' }}>#{index + 1}</td>
                    <td style={{ padding: '8px', color: entry.color, fontWeight: 'bold' }}>{entry.label}</td>
                    <td style={{ padding: '8px' }}>{entry.score.toFixed(4)}</td>
                    <td style={{ padding: '8px' }}>
                      <SaveArchButton 
                        archId={entry.arch_id} 
                        defaultName={entry.label.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase()} 
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </div>
  );
}