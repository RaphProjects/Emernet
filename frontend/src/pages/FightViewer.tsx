import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';
import { useState, useMemo, useEffect } from 'react';
import SaveArchButton from '../components/SaveArchButton';

interface Snapshot {
  epoch: number;
  loss: number | null;
  y: number[];
}

interface FightSide {
  label: string;
  arch_id: string;
  target: number[];
  snapshots: Snapshot[];
  loss_history: number[];
  broken: boolean;
  fit_time: number;  
  score: number;   
}

interface FightData {
  x: number[];
  fight_a: FightSide;
  fight_b: FightSide;
}

function formatChartData(xCoords: number[], fightSide: FightSide, snapshotIndex: number) {
  if (!fightSide.snapshots || fightSide.snapshots.length === 0) return [];
  const safeIdx = Math.min(snapshotIndex, fightSide.snapshots.length - 1);
  const currentSnap = fightSide.snapshots[safeIdx];

  return xCoords.map((xVal, i) => ({
    x: Number(xVal.toFixed(2)),
    target: fightSide.target[i],
    pred: Math.max(-2.5, Math.min(2.5, currentSnap.y[i])),
  }));
}

function FightChart({ title, fightSide, fightData, snapIdx, isWinner }: {
  title: string;
  fightSide: FightSide;
  fightData: FightData;
  snapIdx: number;
  isWinner: boolean;
}) {
  const [autoScaleY, setAutoScaleY] = useState<boolean>(false);

  const currentLoss = fightSide.snapshots[snapIdx]?.loss;
  const chartData = useMemo(
    () => formatChartData(fightData.x, fightSide, snapIdx),
    [fightData.x, fightSide, snapIdx]
  );

  return (
    <div style={{ 
      backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px', 
      display: 'flex', flexDirection: 'column', gap: '4px',
      border: isWinner ? '2px solid #fbbf24' : '2px solid transparent' 
    }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
        <div>
          <h3 style={{ color: isWinner ? '#fbbf24' : 'white', margin: 0, fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            {isWinner && <span>Winner: </span>} {title}
          </h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', columnGap: '12px', marginTop: '6px', fontSize: '12px', color: '#94a3b8' }}>
            <span>Loss:</span> <strong style={{ color: 'white' }}>{currentLoss != null ? currentLoss.toFixed(4) : 'NaN'}</strong>
            <span>Score:</span> <strong style={{ color: '#10b981' }}>{fightSide.score.toFixed(2)}</strong>
            <span>Time:</span> <strong style={{ color: 'white' }}>{fightSide.fit_time.toFixed(2)}s</strong>
          </div>

          <button 
            onClick={() => setAutoScaleY(!autoScaleY)}
            style={{ 
              marginTop: '8px', padding: '2px 8px', fontSize: '11px', 
              background: autoScaleY ? '#3b82f6' : '#334155', 
              color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' 
            }}
          >
            {autoScaleY ? 'Free Y-Axis' : 'Locked Y-Axis'}
          </button>
        </div>

        <SaveArchButton archId={fightSide.arch_id} defaultName={title.replace(/\s+/g, '_')} />
      </div>

      {fightSide.broken ? (
        <div style={{ color: '#ef4444', textAlign: 'center', marginTop: '80px', fontSize: '20px', fontWeight: 'bold' }}>
          ARCH BROKEN (NaN)
        </div>
      ) : (
        <div style={{ height: '320px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="x" stroke="#94a3b8" tickFormatter={(v: number) => v.toFixed(1)} interval={24} />
              <YAxis stroke="#94a3b8" domain={[-2, 2]} ticks={[-2, -1, 0, 1, 2]} tickFormatter={(v: number) => v.toFixed(1)} allowDataOverflow={true} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', color: '#fff', fontSize: '12px' }} />
              <Line type="monotone" name="Target" dataKey="target" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" name="Learner" dataKey="pred" stroke="#f97316" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default function FightViewer() {
  const [pklFiles, setPklFiles] = useState<string[]>([]);
  const [archAFile, setArchAFile] = useState<string>("");
  const [archBFile, setArchBFile] = useState<string>("");
  const [fightData, setFightData] = useState<FightData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [snapIdx, setSnapIdx] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const fetchFiles = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/saved_archs");
      if (res.ok) {
        const data = await res.json();
        setPklFiles(data.files || []);
      }
    } catch (err) {
      console.error("Could not load pkl files", err);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleStartFight = async () => {
    setIsLoading(true);
    setFightData(null);
    setError(null);
    setSnapIdx(0);

    try {
      let url = 'http://127.0.0.1:8000/api/fight_viz?';
      if (archAFile) url += `arch_a_file=${encodeURIComponent(archAFile)}&`;
      if (archBFile) url += `arch_b_file=${encodeURIComponent(archBFile)}`;

      const response = await fetch(url);
      
      if (!response.ok) {
        const errData = await response.json();
        setError(`Error: ${errData.detail || response.status}`);
        return;
      }
      const data = await response.json();
      setFightData(data);
    } catch (err) {
      setError('Could not reach the server');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page-content">
      <div className="page-toolbar" style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', alignItems: 'center' }}>
        <h2 style={{ fontSize: '18px', color: '#f1f5f9', margin: 0 }}>
          Fight Arena
        </h2>
        
        <div style={{ width: '1px', height: '24px', background: '#334155' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '13px', color: '#94a3b8' }}>Arch A:</span>
          <select value={archAFile} onChange={(e) => setArchAFile(e.target.value)} style={{ padding: '6px', borderRadius: '4px', background: '#0f172a', color: 'white', border: '1px solid #475569', fontSize: '13px' }}>
            <option value="">Random Generated</option>
            {pklFiles.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>

        <span style={{ fontSize: '14px', color: '#475569', fontWeight: 'bold' }}>VS</span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '13px', color: '#94a3b8' }}>Arch B:</span>
          <select value={archBFile} onChange={(e) => setArchBFile(e.target.value)} style={{ padding: '6px', borderRadius: '4px', background: '#0f172a', color: 'white', border: '1px solid #475569', fontSize: '13px' }}>
            <option value="">Random Generated</option>
            {pklFiles.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          
          <button onClick={fetchFiles} title="Refresh list" style={{ background: '#334155', border: '1px solid #475569', borderRadius: '4px', padding: '5px 10px', marginLeft: '4px', cursor: 'pointer', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            Refresh
          </button>
        </div>

        <div style={{ width: '1px', height: '24px', background: '#334155' }} />

        <button className="btn btn-primary" onClick={handleStartFight} disabled={isLoading}>
          {isLoading ? 'Computing fight (takes 30-60s)...' : 'Start Fight'}
        </button>

        {error && <span style={{ color: '#ef4444', fontSize: '13px' }}>Error: {error}</span>}
      </div>

      <div className="graph-container" style={{ padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>

        {fightData && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', backgroundColor: '#1e293b', padding: '12px 20px', borderRadius: '8px', color: 'white' }}>
            <label style={{ fontWeight: 600, whiteSpace: 'nowrap' }}>Epoch Snapshot</label>
            <input type="range" min="0" max={fightData.fight_a.snapshots.length - 1} value={snapIdx} onChange={(e) => setSnapIdx(Number(e.target.value))} style={{ flex: 1, cursor: 'pointer' }} />
            <span style={{ fontWeight: 700, minWidth: '80px', textAlign: 'right' }}>
              Epoch {fightData.fight_a.snapshots[snapIdx]?.epoch}
            </span>
          </div>
        )}

        {fightData && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <FightChart title={fightData.fight_a.label} fightSide={fightData.fight_a} fightData={fightData} snapIdx={snapIdx} isWinner={fightData.fight_a.score > fightData.fight_b.score} />
            <FightChart title={fightData.fight_b.label} fightSide={fightData.fight_b} fightData={fightData} snapIdx={snapIdx} isWinner={fightData.fight_b.score > fightData.fight_a.score} />
          </div>
        )}

      </div>
    </div>
  );
}