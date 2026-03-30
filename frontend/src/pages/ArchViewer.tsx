import { useEffect, useState, useCallback, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
} from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';

const API = import.meta.env.VITE_API_BASE_URL;

const MODULE_COLORS: Record<string, string> = {
  Input:              '#22c55e',
  LearnableParameter: '#f97316',
  Add:                '#60a5fa',
  MatMul:             '#60a5fa',
  Mult:               '#60a5fa',
  Activation:         '#f87171',
  Normalizer:         '#a78bfa',
  Pooling:            '#facc15',
  Concat:             '#f0abfc',
  Softmax:            '#fb7185',
  Transpose:          '#94a3b8',
  Shift:              '#94a3b8',
  Split:              '#fbbf24',
  EMA:                '#34d399',
  Accumulator:        '#34d399',
  Memory:             '#2dd4bf',
};

function getNodeStyle(type: string): React.CSSProperties {
  return {
    background: MODULE_COLORS[type] ?? '#e2e8f0',
    border: '1px solid #47556988',
    borderRadius: '4px',
    padding: '3px 8px',
    fontSize: '11px',
    fontWeight: 600,
    color: '#0f172a',
    lineHeight: '1.2',
    whiteSpace: 'nowrap' as const,
  };
}

export default function ArchViewer() {
  const [currentArchId, setCurrentArchId] = useState<string | null>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadedFileName, setLoadedFileName] = useState<string>('');
  const [downloadName, setDownloadName] = useState('architecture');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatGraphData = (data: any) => {
    const formattedNodes: Node[] = data.nodes.map((n: any) => ({
      id: String(n.id),
      position: { x: n.x, y: n.y },
      data: { label: n.type },
      style: getNodeStyle(n.type),
    }));

    const formattedEdges: Edge[] = data.edges.map((e: any, i: number) => ({
      id: `e${i}`,
      source: String(e.source),
      target: String(e.target),
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748b', strokeWidth: 1.5 },
      markerEnd: { type: 'arrowclosed' as const, color: '#64748b', width: 14, height: 14 },
    }));

    setNodes(formattedNodes);
    setEdges(formattedEdges);
  };

  /* ── Generate random ───────────────────────────────────── */
  const generateArch = useCallback(async () => {
    setLoading(true);
    setError(null);
    setLoadedFileName('');
    try {
      const res = await fetch(`${API}/api/generate`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setCurrentArchId(data.arch_id);
      setDownloadName('random_arch');
      formatGraphData(data);
    } catch {
      setError('Backend not reachable. Is the server running?');
    } finally {
      setLoading(false);
    }
  }, []);

  /* ── Upload .pkl from user's machine ───────────────────── */
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setLoadedFileName(file.name);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API}/api/upload_arch`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setCurrentArchId(data.arch_id);
      setDownloadName(file.name.replace(/\.pkl$/i, ''));
      formatGraphData(data);
    } catch (err: any) {
      setError(`Failed to load ${file.name}: ${err.message}`);
    } finally {
      setLoading(false);
      // reset so re-uploading the same file still triggers onChange
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  /* ── Download current arch as .pkl ─────────────────────── */
  const downloadArch = async () => {
    if (!currentArchId) return;
    try {
      const res = await fetch(`${API}/api/download_arch/${currentArchId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${downloadName || 'architecture'}.pkl`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError('Failed to download architecture');
    }
  };

  useEffect(() => {
    generateArch();
  }, [generateArch]);

  return (
    <div className="page-content">
      <div className="page-toolbar" style={{ display: 'flex', gap: '15px', flexWrap: 'wrap', alignItems: 'center' }}>

        {/* ── Generate ──────────────────────────── */}
        <button className="btn btn-primary" onClick={generateArch} disabled={loading}>
          {loading && !loadedFileName ? 'Generating...' : 'Generate Random'}
        </button>

        <div style={{ width: '1px', height: '24px', background: '#334155' }} />

        {/* ── Upload ────────────────────────────── */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pkl"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        <button
          className="btn btn-back"
          onClick={() => fileInputRef.current?.click()}
          disabled={loading}
          title="Upload a .pkl architecture file"
          style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          {/* upload icon */}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          {loading && loadedFileName ? 'Loading...' : 'Load .pkl'}
        </button>

        {/* ── Download ──────────────────────────── */}
        {currentArchId && (
          <>
            <div style={{ width: '1px', height: '24px', background: '#334155' }} />

            <input
              type="text"
              value={downloadName}
              onChange={(e) => setDownloadName(e.target.value)}
              placeholder="filename"
              style={{
                background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155',
                padding: '5px 10px', borderRadius: '6px', width: '150px',
                fontSize: '13px', outline: 'none',
              }}
            />
            <button
              className="btn btn-primary"
              onClick={downloadArch}
              title="Download architecture as .pkl"
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              {/* download icon */}
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Save .pkl
            </button>
          </>
        )}

        {/* ── Info / Error ──────────────────────── */}
        {nodes.length > 0 && (
          <span className="toolbar-info" style={{ marginLeft: 'auto' }}>
            {loadedFileName ? `Loaded: ${loadedFileName} ` : ''}
            (Nodes: {nodes.length} | Edges: {edges.length})
          </span>
        )}
        {error && <span className="toolbar-error">{error}</span>}
      </div>

      <div className="graph-container">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.4, maxZoom: 1.5 }}
          proOptions={{ hideAttribution: true }}
          minZoom={0.2}
          maxZoom={3}
          nodesDraggable={true}
          nodesConnectable={false}
          elementsSelectable={true}
        >
          <Background gap={24} size={1} color="#1e293b" />
          <Controls showInteractive={false} position="bottom-right" />
          <MiniMap
            nodeColor={(n) => {
              const type = n.data?.label as string;
              return MODULE_COLORS[type] ?? '#e2e8f0';
            }}
            style={{ background: '#1e293b', border: '1px solid #334155' }}
            maskColor="rgba(15, 23, 42, 0.7)"
          />
        </ReactFlow>
      </div>
    </div>
  );
}