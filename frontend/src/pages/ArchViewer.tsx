import { useEffect, useState, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
} from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';
import SaveArchButton from '../components/SaveArchButton';


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

  // New State for loading files
  const [savedFiles, setSavedFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');

  // Helper to format raw JSON into ReactFlow format
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

  // Fetch random architecture
  const generateArch = useCallback(async () => {
    setLoading(true);
    setError(null);
    setSelectedFile('');
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/generate`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      // Parse JSON first
      setCurrentArchId(data.arch_id); 
      // Then read from parsed data
      formatGraphData(data); 
      // Then format the graph
    } catch (err: any) {
      setError('Backend not reachable. Is the server running?');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch specific saved architecture
  const loadArch = async (filename: string) => {
    if (!filename) return;
    setLoading(true);
    setError(null);
    setSelectedFile(filename);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/load_arch/${filename}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      if (data.error) throw new Error(data.error);
      formatGraphData(data);
    } catch (err: any) {
      setError(`Failed to load ${filename}`);
    } finally {
      setLoading(false);
    }
  };

  // Get the list of .pkl files
  const fetchFileList = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/saved_archs');
      const data = await res.json();
      if (data.files) setSavedFiles(data.files);
    } catch (err) {
      console.error("Could not fetch file list");
    }
  };

  // Run on first load
  useEffect(() => {
    generateArch();
    fetchFileList();
  }, [generateArch]);

  return (
    <div className="page-content">
      <div className="page-toolbar" style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
        
        <button className="btn btn-primary" onClick={generateArch} disabled={loading}>
          {loading && !selectedFile ? 'Generating...' : 'Generate Random'}
        </button>

        {/* Add the Save Button here */}
        {currentArchId && (
          <>
            <div style={{ width: '1px', height: '24px', background: '#334155', margin: '0 8px' }} />
            <SaveArchButton archId={currentArchId} defaultName="random_arch" />
          </>
        )}

        {/* The Dropdown Menu for Saved Files */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select 
            value={selectedFile}
            onChange={(e) => loadArch(e.target.value)}
            style={{ 
              background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155', 
              padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', outline: 'none'
            }}
          >
            <option value="">Load Saved Arch</option>
            {savedFiles.map(file => (
              <option key={file} value={file}>{file}</option>
            ))}
          </select>
          <button className="btn btn-back" onClick={fetchFileList} title="Refresh File List" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10"></polyline>
              <polyline points="1 20 1 14 7 14"></polyline>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
            </svg>
          </button>
        </div>

        {nodes.length > 0 && (
          <span className="toolbar-info" style={{ marginLeft: 'auto' }}>
            {selectedFile ? `Loaded: ${selectedFile} ` : ''} 
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