import { useEffect, useState, useCallback, useRef } from 'react';
import type React from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  useReactFlow,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
} from '@xyflow/react';
import type { Node, Edge, Connection, OnNodesDelete, OnEdgesDelete, OnEdgesChange, OnNodesChange } from '@xyflow/react';
import {
  generateArch, uploadArch, downloadArch as downloadArchApi, downloadArchPython,
  importArchSubgraph, uploadPolicy, predictArchWithPolicy,
} from '../api';
import { MODULE_COLORS, MODULE_SET_OPTIONS } from '../theme';
import NodeInspector from '../components/NodeInspector';
import NodePalette from '../components/NodePalette';
import LooseNumberInput from '../components/LooseNumberInput';

const API = import.meta.env.VITE_API_BASE_URL;
const DEBUG_ARCH_MUTATIONS = true;

function getNodeStyle(type: string, pending?: boolean): React.CSSProperties {
  const color = MODULE_COLORS[type] ?? '#e2e8f0';
  return {
    background: pending ? `color-mix(in srgb, ${color} 20%, rgba(0,0,0,0.84))` : `${color}e6`,
    border: pending ? `2px dashed ${color}` : '1px solid rgba(0, 0, 0, 0.55)',
    borderRadius: '8px',
    padding: '5px 12px',
    fontSize: '11px',
    fontWeight: 800,
    color: pending ? 'var(--text-secondary)' : '#100800',
    lineHeight: '1.3',
    whiteSpace: 'nowrap' as const,
    fontFamily: 'inherit',
    boxShadow: pending ? '0 0 8px rgba(var(--theme-accent-rgb),0.18)' : '0 2px 8px rgba(0,0,0,0.28)',
    transition: 'box-shadow 0.2s, border-color 0.2s, background 0.2s',
  };
}

function getEdgeStyle(userAdded: boolean): React.CSSProperties {
  return {
    stroke: userAdded ? 'rgba(var(--theme-accent-rgb), 0.72)' : 'rgba(var(--theme-primary-rgb), 0.5)',
    strokeWidth: userAdded ? 2.5 : 1.5,
    strokeDasharray: userAdded ? '5 3' : undefined,
  };
}

interface PendingChange {
  action: 'add_node' | 'remove_node' | 'add_edge' | 'remove_edge' | 'modify_params';
  params: Record<string, any>;
}

function formatGraphData(data: any) {
  const formattedNodes: Node[] = data.nodes.map((n: any) => ({
    id: String(n.id),
    position: { x: n.x, y: n.y },
    data: { label: n.type, module_type: n.module_type },
    style: getNodeStyle(n.type),
  }));
  const formattedEdges: Edge[] = data.edges.map((e: any, i: number) => ({
    id: `e${i}`,
    source: String(e.source),
    target: String(e.target),
    type: 'default',
    animated: false,
    style: getEdgeStyle(false),
    markerEnd: { type: 'arrowclosed' as const, color: 'rgba(255,122,24,0.5)', width: 14, height: 14 },
  }));
  return { nodes: formattedNodes, edges: formattedEdges };
}

function ArchViewerInner() {
  const [currentArchId, setCurrentArchId] = useState<string | null>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadedFileName, setLoadedFileName] = useState<string>('');
  const [downloadName, setDownloadName] = useState('architecture');
  const [moduleSet, setModuleSet] = useState('Unified');
  const [archSize, setArchSize] = useState(12);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [showPalette, setShowPalette] = useState(false);
  const [pendingChanges, setPendingChanges] = useState<PendingChange[]>([]);
  const [pendingParamEdits, setPendingParamEdits] = useState<Record<number, Record<string, any>>>({});
  const [applying, setApplying] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const policyInputRef = useRef<HTMLInputElement>(null);
  const selectedEdgeIdRef = useRef<string | null>(null);
  const pendingIdCounter = useRef(0);
  const reactFlowInstance = useReactFlow();
  const [policyModel, setPolicyModel] = useState<{ id: string; name: string; records?: number } | null>(null);
  const [policyPrediction, setPolicyPrediction] = useState<any>(null);
  const [policyBusy, setPolicyBusy] = useState(false);
  const [policyOppBal, setPolicyOppBal] = useState(0.2);
  const [policySpeedBal, setPolicySpeedBal] = useState(0.3);

  const selectedEdge = edges.find(e => e.id === selectedEdgeIdRef.current && e.selected) ?? null;
  const hasChanges = pendingChanges.length > 0 || Object.keys(pendingParamEdits).length > 0;

  const undoStack = useRef<{ nodes: Node[]; edges: Edge[]; archId: string | null }[]>([]);
  const redoStack = useRef<{ nodes: Node[]; edges: Edge[]; archId: string | null }[]>([]);
  const currentState = useRef<{ nodes: Node[]; edges: Edge[]; archId: string | null }>({ nodes: [], edges: [], archId: null });

  const pushState = useCallback((n: Node[], e: Edge[], id: string | null) => {
    undoStack.current.push(JSON.parse(JSON.stringify({ nodes: n, edges: e, archId: id })));
    redoStack.current = [];
    currentState.current = { nodes: n, edges: e, archId: id };
  }, []);

  const loadArchData = useCallback((data: any) => {
    const { nodes: fmtNodes, edges: fmtEdges } = formatGraphData(data);
    setNodes(fmtNodes);
    setEdges(fmtEdges);
    setCurrentArchId(data.arch_id);
    setPendingChanges([]);
    setPendingParamEdits({});
    setPolicyPrediction(null);
    selectedEdgeIdRef.current = null;
    pushState(fmtNodes, fmtEdges, data.arch_id);
    setTimeout(() => reactFlowInstance.fitView({ padding: 0.4, maxZoom: 1.5 }), 100);
  }, [reactFlowInstance, pushState]);

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setLoadedFileName('');
    try {
      const data = await generateArch(moduleSet, archSize);
      setDownloadName('random_arch');
      loadArchData(data);
    } catch {
      setError('Backend not reachable. Is the server running?');
    } finally {
      setLoading(false);
    }
  }, [moduleSet, archSize, loadArchData]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    setLoadedFileName(file.name);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API}/api/upload_arch`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setDownloadName(file.name.replace(/\.pkl$/i, ''));
      loadArchData(data);
    } catch (err: any) {
      setError(`Failed to load ${file.name}: ${err.message}`);
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handlePolicyUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPolicyBusy(true);
    setError(null);
    try {
      const data = await uploadPolicy(file);
      setPolicyModel({ id: data.policy_id, name: file.name, records: data.records_seen });
      setPolicyPrediction(null);
    } catch (err: any) {
      setError(`Failed to load policy: ${err.message}`);
    } finally {
      setPolicyBusy(false);
      if (policyInputRef.current) policyInputRef.current.value = '';
    }
  };

  const predictCurrentArch = async () => {
    if (!currentArchId || !policyModel) return;
    setPolicyBusy(true);
    setError(null);
    try {
      const data = await predictArchWithPolicy({
        arch_id: currentArchId,
        policy_id: policyModel.id,
        module_set: moduleSet,
        speed_bal: policySpeedBal,
        opp_simp_bal: policyOppBal,
      });
      setPolicyPrediction(data);
    } catch (err: any) {
      setError(`Policy prediction failed: ${err.message}`);
    } finally {
      setPolicyBusy(false);
    }
  };

  const handleImportArch = useCallback(async (file: File) => {
    if (!currentArchId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await importArchSubgraph(currentArchId, file);
      loadArchData(data);
    } catch (err: any) {
      setError(`Import failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [currentArchId, loadArchData]);

  const handleDownload = async () => {
    if (!currentArchId) return;
    try {
      await downloadArchApi(currentArchId, downloadName || 'architecture');
    } catch {
      setError('Failed to download architecture');
    }
  };

  const handlePythonExport = async () => {
    if (!currentArchId) return;
    try {
      await downloadArchPython(currentArchId, `${downloadName || 'architecture'}_model`);
    } catch (err: any) {
      setError(`Python export failed: ${err.message ?? 'unknown error'}`);
    }
  };

  const queueChange = useCallback((change: PendingChange) => {
    if (DEBUG_ARCH_MUTATIONS) {
      console.debug('[ArchViewer] queueChange', change);
    }
    setPolicyPrediction(null);
    setPendingChanges(prev => [...prev, change]);
  }, []);

  const nextPendingId = useCallback((prefix: string) => {
    pendingIdCounter.current += 1;
    return `${prefix}-${Date.now()}-${pendingIdCounter.current}`;
  }, []);

  const onNodesChange: OnNodesChange = useCallback((changes) => {
    setNodes(prev => applyNodeChanges(changes, prev));
  }, []);

  const onEdgesChange: OnEdgesChange = useCallback((changes) => {
    for (const c of changes) {
      if (c.type === 'select') {
        selectedEdgeIdRef.current = c.selected ? (c as any).id : null;
        setEdges(prev => applyEdgeChanges(changes, prev));
        break;
      }
    }
  }, []);

  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target) return;
    const source = connection.source;
    const target = connection.target;
    const sourceNode = nodes.find(n => n.id === source);
    const targetNode = nodes.find(n => n.id === target);
    if (DEBUG_ARCH_MUTATIONS) {
      console.groupCollapsed('[ArchViewer] onConnect');
      console.debug('raw connection', connection);
      console.debug('source node', sourceNode);
      console.debug('target node', targetNode);
      console.debug('current nodes', nodes.map(n => ({
        id: n.id,
        label: n.data?.label,
        position: n.position,
      })));
      console.debug('current edges', edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
      })));
      console.groupEnd();
    }
    if (source === target) {
      console.warn('[ArchViewer] blocked self-edge from React Flow', { source, target, connection });
      setError(`Blocked self-edge on ${source}. Check browser console for connection details.`);
      return;
    }
    queueChange({ action: 'add_edge', params: { source, target } });
    setEdges(prev => [...prev, {
      id: nextPendingId('pending-e'),
      source: String(source),
      target: String(target),
      type: 'default',
      style: getEdgeStyle(true),
      markerEnd: { type: 'arrowclosed' as const, color: 'rgba(255,242,61,0.72)', width: 14, height: 14 },
    }]);
    setError(null);
  }, [edges, nextPendingId, nodes, queueChange]);

  const onNodesDelete: OnNodesDelete = useCallback((deleted) => {
    for (const node of deleted) {
      if (node.id.startsWith('pending-')) continue;
      queueChange({ action: 'remove_node', params: { node_id: parseInt(node.id) } });
    }
    setNodes(prev => prev.filter(n => !deleted.find(d => d.id === n.id)));
    setEdges(prev => prev.filter(e =>
      !deleted.find(d => d.id === e.source || d.id === e.target)
    ));
    setError(null);
  }, [queueChange]);

  const onEdgesDelete: OnEdgesDelete = useCallback((deleted) => {
    for (const edge of deleted) {
      const source = parseInt(edge.source);
      const target = parseInt(edge.target);
      if (isNaN(source) || isNaN(target)) continue;
      queueChange({
        action: 'remove_edge',
        params: { source, target },
      });
    }
    setEdges(prev => prev.filter(e => !deleted.find(d => d.id === e.id)));
    setError(null);
  }, [queueChange]);

  const onNodeClick = useCallback((_: any, node: Node) => {
    if (node.id.startsWith('pending-')) return;
    setSelectedNode(parseInt(node.id));
  }, []);

  const handleAddNode = useCallback((moduleType: string) => {
    const pendingId = nextPendingId('pending-node');
    if (DEBUG_ARCH_MUTATIONS) {
      console.debug('[ArchViewer] add pending node', { pendingId, moduleType });
    }
    queueChange({ action: 'add_node', params: { module_type: moduleType, client_id: pendingId } });
    const newNode: Node = {
      id: pendingId,
      position: { x: 250 + Math.random() * 200, y: 150 + Math.random() * 150 },
      data: { label: moduleType },
      style: getNodeStyle(moduleType, true),
    };
    setNodes(prev => [...prev, newNode]);
    setError(null);
  }, [nextPendingId, queueChange]);

  const handleParamChange = useCallback((nodeId: number, key: string, value: any) => {
    setPendingParamEdits(prev => ({
      ...prev,
      [nodeId]: { ...(prev[nodeId] || {}), [key]: value },
    }));
  }, []);

  const handleApply = async () => {
    if (!currentArchId || (!hasChanges)) return;
    setApplying(true);
    setError(null);

    const mutations: PendingChange[] = [...pendingChanges];

    for (const [nodeIdStr, params] of Object.entries(pendingParamEdits)) {
      const nodeId = parseInt(nodeIdStr);
      if (isNaN(nodeId)) continue;
      mutations.push({ action: 'modify_params', params: { node_id: nodeId, params } });
    }

    if (DEBUG_ARCH_MUTATIONS) {
      console.groupCollapsed('[ArchViewer] apply mutations');
      console.debug('arch_id', currentArchId);
      console.debug('mutations', mutations);
      console.debug('nodes', nodes.map(n => ({
        id: n.id,
        label: n.data?.label,
        position: n.position,
      })));
      console.debug('edges', edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
      })));
      console.groupEnd();
    }

    try {
      const res = await fetch(`${API}/api/mutate/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ arch_id: currentArchId, mutations }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Mutation failed');
      }
      const data = await res.json();
      loadArchData(data);
    } catch (err: any) {
      setError(`Apply failed: ${err.message}`);
    } finally {
      setApplying(false);
    }
  };

  const handleUndo = useCallback(() => {
    const prev = undoStack.current.pop();
    if (!prev) return;
    redoStack.current.push(currentState.current);
    setNodes(prev.nodes);
    setEdges(prev.edges);
    setCurrentArchId(prev.archId);
    setPendingChanges([]);
    setPendingParamEdits({});
    setSelectedNode(null);
    currentState.current = prev;
  }, []);

  const handleRedo = useCallback(() => {
    const next = redoStack.current.pop();
    if (!next) return;
    undoStack.current.push(currentState.current);
    setNodes(next.nodes);
    setEdges(next.edges);
    setCurrentArchId(next.archId);
    setPendingChanges([]);
    setPendingParamEdits({});
    setSelectedNode(null);
    currentState.current = next;
  }, []);

  const discardChanges = useCallback(() => {
    const state = currentState.current;
    setNodes(state.nodes);
    setEdges(state.edges);
    setCurrentArchId(state.archId);
    setPendingChanges([]);
    setPendingParamEdits({});
    setSelectedNode(null);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z') {
        e.preventDefault();
        handleRedo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        handleUndo();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo]);

  useEffect(() => {
    handleGenerate();
  }, []);

  return (
    <div className="page-content">
      <div className="page-toolbar" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
        <button className="btn btn-primary" onClick={handleGenerate} disabled={loading || applying}>
          {loading ? 'Generating...' : 'Generate'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 700 }}>Modules:</label>
          <select value={moduleSet} onChange={e => setModuleSet(e.target.value)}
            style={{
              padding: '4px 8px', fontSize: '12px', outline: 'none', cursor: 'pointer',
            }}>
            {MODULE_SET_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 700 }}>Nodes:</label>
          <LooseNumberInput
            min={3}
            max={64}
            value={archSize}
            onChange={setArchSize}
            fallback={3}
            style={{ width: '64px', padding: '4px 8px', fontSize: '12px' }}
            title="Target number of generated non-input nodes"
          />
        </div>

        <div style={{ width: '1px', height: '20px', background: 'var(--glass-border)' }} />

        <input ref={fileInputRef} type="file" accept=".pkl" onChange={handleFileUpload} style={{ display: 'none' }} />
        <input ref={policyInputRef} type="file" accept=".pkl" onChange={handlePolicyUpload} style={{ display: 'none' }} />
        <button className="btn btn-back" onClick={() => fileInputRef.current?.click()} disabled={loading}
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          Load
        </button>
        <button className="btn btn-back" onClick={() => policyInputRef.current?.click()} disabled={policyBusy}
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
          {policyModel ? `Policy: ${policyModel.name}` : 'Load Policy'}
        </button>

        {currentArchId && (
          <>
            <input type="text" value={downloadName} onChange={e => setDownloadName(e.target.value)}
              placeholder="filename"
              style={{
                padding: '4px 10px', width: '130px', fontSize: '12px', outline: 'none',
              }}
            />
            <button className="btn btn-back" onClick={handleDownload}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Save
            </button>
            <button className="btn btn-back" onClick={handlePythonExport}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
              Export .py
            </button>
          </>
        )}

        <div style={{ width: '1px', height: '20px', background: 'var(--glass-border)' }} />
        {policyModel && currentArchId && (
          <>
            <label style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 700 }}>Speed:</label>
            <LooseNumberInput
              step="0.05"
              value={policySpeedBal}
              onChange={setPolicySpeedBal}
              style={{ width: '64px', padding: '4px 8px', fontSize: '12px' }}
            />
            <label style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 700 }}>Opp:</label>
            <LooseNumberInput
              step="0.05"
              value={policyOppBal}
              onChange={setPolicyOppBal}
              style={{ width: '64px', padding: '4px 8px', fontSize: '12px' }}
            />
            <button className="btn btn-apply" onClick={predictCurrentArch} disabled={policyBusy || hasChanges}
              style={{ fontSize: '12px' }}>
              {policyBusy ? 'Predicting...' : 'Predict Arch'}
            </button>
          </>
        )}
        <div style={{ width: '1px', height: '20px', background: 'var(--glass-border)' }} />
        <button className="btn btn-back" onClick={() => setShowPalette(p => !p)} style={{ fontSize: '12px' }}>
          {showPalette ? 'Close' : 'Add Node'}
        </button>

        <button className="btn btn-back" onClick={handleUndo} disabled={undoStack.current.length === 0 || applying}
          style={{ fontSize: '12px', padding: '4px 10px' }} title="Undo (Ctrl+Z)">↩</button>
        <button className="btn btn-back" onClick={handleRedo} disabled={redoStack.current.length === 0 || applying}
          style={{ fontSize: '12px', padding: '4px 10px' }} title="Redo (Ctrl+Shift+Z)">↪</button>

        {selectedEdge && (
          <button className="btn btn-back" onClick={() => {
            onEdgesDelete([selectedEdge]);
          }} style={{ fontSize: '12px', color: '#fca5a5' }}>
            Delete Edge
          </button>
        )}

        {hasChanges && (
          <span className="change-indicator">{pendingChanges.length + Object.keys(pendingParamEdits).length} change(s)</span>
        )}

        <div style={{ flex: 1 }} />

        {hasChanges && (
          <>
            <button className="btn btn-back" onClick={discardChanges} disabled={applying}
              style={{ fontSize: '12px', color: '#fca5a5' }}>
              Discard
            </button>
            <button
              className={`btn btn-apply ${hasChanges ? 'has-changes' : ''}`}
              onClick={handleApply} disabled={applying || !hasChanges}
              style={{ fontSize: '12px' }}
            >
              {applying ? 'Applying...' : `Apply Changes (${pendingChanges.length + Object.keys(pendingParamEdits).length})`}
            </button>
          </>
        )}

        {nodes.length > 0 && !hasChanges && (
          <span className="toolbar-info">
            {nodes.length} nodes · {edges.length} edges
          </span>
        )}
        {policyPrediction && !hasChanges && (
          <span className="toolbar-info">
            Lrn {fmt(policyPrediction.learnability)} · Spd {fmt(policyPrediction.speed)} · Opp {fmt(policyPrediction.opp_simp_raw)} · Final {fmt(policyPrediction.final_score)}
          </span>
        )}
        {error && <span className="toolbar-error">{error}</span>}
      </div>

      <div className="graph-container" style={{ position: 'relative' }}>
        {showPalette && (
          <NodePalette onAddNode={handleAddNode} onClose={() => setShowPalette(false)} onImportArch={handleImportArch} />
        )}

        {selectedNode !== null && currentArchId && (
          <NodeInspector
            archId={currentArchId}
            nodeId={selectedNode}
            pendingChanges={pendingParamEdits[selectedNode] || {}}
            onParamChange={handleParamChange}
            onClose={() => setSelectedNode(null)}
          />
        )}

        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodesDelete={onNodesDelete}
          onEdgesDelete={onEdgesDelete}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{ padding: 0.4, maxZoom: 1.5 }}
          proOptions={{ hideAttribution: true }}
          minZoom={0.2}
          maxZoom={3}
          nodesDraggable={true}
          nodesConnectable={true}
          elementsSelectable={true}
          deleteKeyCode={['Backspace', 'Delete']}
          connectionLineStyle={{ stroke: 'rgba(251,191,36,0.5)', strokeWidth: 2 }}
          defaultEdgeOptions={{
            type: 'default',
            style: getEdgeStyle(false),
            markerEnd: { type: 'arrowclosed' as const, color: 'rgba(255,122,24,0.5)', width: 14, height: 14 },
          }}
        >
          <Background gap={28} size={1} color="rgba(255, 122, 24, 0.18)" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </div>
    </div>
  );
}

function fmt(value: any, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
}

export default function ArchViewer() {
  return (
    <ReactFlowProvider>
      <ArchViewerInner />
    </ReactFlowProvider>
  );
}
