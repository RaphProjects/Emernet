import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';
import { useState, useMemo, useRef } from 'react';

const API = import.meta.env.VITE_API_BASE_URL;

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

/* ── Download helper ─────────────────────────────────────── */
async function downloadArch(archId: string, filename: string) {
  try {
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
  } catch (err) {
    console.error('Download failed:', err);
  }
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

        {/* ── Download button (replaces SaveArchButton) ── */}
        <button
          className="btn btn-primary"
          onClick={() => downloadArch(fightSide.arch_id, title.replace(/\s+/g, '_'))}
          title="Download this architecture as .pkl"
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', padding: '6px 12px' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Save .pkl
        </button>
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
  const [archAId, setArchAId] = useState<string | null>(null);
  const [archBId, setArchBId] = useState<string | null>(null);
  const [archAName, setArchAName] = useState<string>('');
  const [archBName, setArchBName] = useState<string>('');
  const [uploadingA, setUploadingA] = useState(false);
  const [uploadingB, setUploadingB] = useState(false);
  const [fightData, setFightData] = useState<FightData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [snapIdx, setSnapIdx] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const fileInputA = useRef<HTMLInputElement>(null);
  const fileInputB = useRef<HTMLInputElement>(null);

  /* ── Upload a .pkl and get back an arch_id ─────────────── */
  const handleUpload = async (
    file: File,
    setId: (id: string | null) => void,
    setName: (n: string) => void,
    setUploading: (b: boolean) => void,
    inputRef: React.RefObject<HTMLInputElement>,
  ) => {
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API}/api/upload_arch`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setId(data.arch_id);
      setName(file.name.replace(/\.pkl$/i, ''));
    } catch (err: any) {
      setError(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  const clearArch = (
    setId: (id: string | null) => void,
    setName: (n: string) => void,
  ) => {
    setId(null);
    setName('');
  };

  /* ── Start fight ───────────────────────────────────────── */
  const handleStartFight = async () => {
    setIsLoading(true);
    setFightData(null);
    setError(null);
    setSnapIdx(0);
    try {
      const params = new URLSearchParams();
      if (archAId) params.set('arch_a_id', archAId);
      if (archBId) params.set('arch_b_id', archBId);

      const res = await fetch(`${API}/api/fight_viz?${params.toString()}`);
      if (!res.ok) {
        const errData = await res.json();
        setError(`Error: ${errData.detail || res.status}`);
        return;
      }
      setFightData(await res.json());
    } catch {
      setError('Could not reach the server');
    } finally {
      setIsLoading(false);
    }
  };

  /* ── Arch slot UI ──────────────────────────────────────── */
  const ArchSlot = ({ label, name, uploading, inputRef, onUpload, onClear }: {
    label: string;
    name: string;
    uploading: boolean;
    inputRef: React.RefObject<HTMLInputElement>;
    onUpload: (f: File) => void;
    onClear: () => void;
  }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ fontSize: '13px', color: '#94a3b8', fontWeight: 600 }}>{label}:</span>

      <input
        ref={inputRef}
        type="file"
        accept=".pkl"
        style={{ display: 'none' }}
        onChange={(e) => { if (e.target.files?.[0]) onUpload(e.target.files[0]); }}
      />

      {name ? (
        <>
          <span style={{
            background: '#0f172a', border: '1px solid #475569', borderRadius: '4px',
            padding: '5px 10px', fontSize: '13px', color: '#f1f5f9', maxWidth: '180px',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {name}.pkl
          </span>
          <button
            onClick={onClear}
            title="Remove — will use random instead"
            style={{
              background: '#7f1d1d', border: '1px solid #991b1b', borderRadius: '4px',
              padding: '4px 8px', cursor: 'pointer', color: '#fca5a5', fontSize: '12px',
            }}
          >✕</button>
        </>
      ) : (
        <button
          className="btn btn-back"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          {uploading ? 'Uploading...' : 'Upload .pkl'}
        </button>
      )}

      {!name && (
        <span style={{ fontSize: '11px', color: '#475569', fontStyle: 'italic' }}>Random</span>
      )}
    </div>
  );

  return (
    <div className="page-content">
      <div className="page-toolbar" style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', alignItems: 'center' }}>
        <h2 style={{ fontSize: '18px', color: '#f1f5f9', margin: 0 }}>Fight Arena</h2>

        <div style={{ width: '1px', height: '24px', background: '#334155' }} />

        <ArchSlot
          label="Arch A" name={archAName} uploading={uploadingA} inputRef={fileInputA}
          onUpload={(f) => handleUpload(f, setArchAId, setArchAName, setUploadingA, fileInputA)}
          onClear={() => clearArch(setArchAId, setArchAName)}
        />

        <span style={{ fontSize: '14px', color: '#475569', fontWeight: 'bold' }}>VS</span>

        <ArchSlot
          label="Arch B" name={archBName} uploading={uploadingB} inputRef={fileInputB}
          onUpload={(f) => handleUpload(f, setArchBId, setArchBName, setUploadingB, fileInputB)}
          onClear={() => clearArch(setArchBId, setArchBName)}
        />

        <div style={{ width: '1px', height: '24px', background: '#334155' }} />

        <button className="btn btn-primary" onClick={handleStartFight} disabled={isLoading}>
          {isLoading ? 'Computing fight (takes 30-60s)...' : 'Start Fight'}
        </button>

        {error && <span style={{ color: '#ef4444', fontSize: '13px' }}>⚠ {error}</span>}
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