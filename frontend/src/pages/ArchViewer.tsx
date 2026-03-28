import { useEffect, useState, useCallback } from 'react';
import { type Node, type Edge, Position, ReactFlow, Controls, Background } from '@xyflow/react';
const MODULE_COLORS: Record<string, string> = {
  Input: '#22c55e',
  LearnableParameter: '#f97316',
  Add: '#60a5fa',
  MatMul: '#60a5fa',
  Mult: '#60a5fa',
  Activation: '#f87171',
  Normalizer: '#a78bfa',
  Pooling: '#facc15',
  Concat: '#f0abfc',
  Softmax: '#fb7185',
  Transpose: '#94a3b8',
  Shift: '#94a3b8',
  EMA: '#34d399',
  Accumulator: '#34d399',
  Split: '#fbbf24',
};

// Compact node style
function getNodeStyle(type: string): React.CSSProperties {
  return {
    background: MODULE_COLORS[type] ?? '#e2e8f0',
    border: '1px solid #1e293b',
    borderRadius: '6px',
    width: '90px',        // Strictly limit width
    height: '32px',       // Strictly limit height
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '10px',
    fontWeight: 700,
    color: '#0f172a',
    padding: '0 4px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
  };
}

// Shorten long names so they fit in the small nodes
function formatName(name: string): string {
  if (name === 'LearnableParameter') return 'Learnable';
  if (name === 'Accumulator') return 'Accum';
  if (name === 'Normalizer') return 'Norm';
  if (name === 'Activation') return 'Activ';
  return name;
}

// Hierarchical Top-to-Bottom Layout Algorithm
function getLayoutedElements(nodes: any[], edges: any[]) {
  const outEdges = new Map<string, string[]>();
  const inDegree = new Map<string, number>();

  // Initialize maps
  nodes.forEach(n => {
    outEdges.set(n.id, []);
    inDegree.set(n.id, 0);
  });

  // Populate edges & degrees
  edges.forEach(e => {
    if (outEdges.has(e.source)) outEdges.get(e.source)!.push(e.target);
    if (inDegree.has(e.target)) inDegree.set(e.target, inDegree.get(e.target)! + 1);
  });

  // Find roots (Input nodes, etc.)
  const queue: string[] = [];
  nodes.forEach(n => {
    if (inDegree.get(n.id) === 0) queue.push(n.id);
  });

  // Calculate layers (depth) via BFS
  const layers = new Map<string, number>();
  queue.forEach(id => layers.set(id, 0));

  while (queue.length > 0) {
    const curr = queue.shift()!;
    const currLayer = layers.get(curr)!;

    outEdges.get(curr)!.forEach(neighbor => {
      // Push node down to the deepest required layer
      layers.set(neighbor, Math.max(layers.get(neighbor) || 0, currLayer + 1));
      inDegree.set(neighbor, inDegree.get(neighbor)! - 1);
      
      if (inDegree.get(neighbor) === 0) {
        queue.push(neighbor);
      }
    });
  }

  // Group nodes by their layer
  const layerGroups: string[][] = [];
  layers.forEach((layerIndex, nodeId) => {
    while (layerGroups.length <= layerIndex) layerGroups.push([]);
    layerGroups[layerIndex].push(nodeId);
  });

  // Assign X and Y coordinates based on layer grid
  const Y_SPACING = 80;  // Vertical space between layers
  const X_SPACING = 110; // Horizontal space between nodes in the same layer

  const layoutedNodes = nodes.map(n => {
    const layerIndex = layers.get(n.id) || 0;
    const group = layerGroups[layerIndex];
    const indexInGroup = group.indexOf(n.id);
    
    // Center the nodes in the layer
    const totalWidth = (group.length - 1) * X_SPACING;
    const startX = -totalWidth / 2;

    return {
      ...n,
      targetPosition: Position.Top,    // Wires enter from top
      sourcePosition: Position.Bottom, // Wires exit from bottom
      position: {
        x: startX + indexInGroup * X_SPACING,
        y: layerIndex * Y_SPACING
      }
    };
  });

  return layoutedNodes;
}

export default function ArchViewer() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [loading, setLoading] = useState(false);

  const generateArch = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/generate');
      const data = await res.json();

      // 1. Create Raw Nodes
      const rawNodes = data.nodes.map((n: any) => ({
        id: n.id.toString(),
        data: { label: formatName(n.type) },
        style: getNodeStyle(n.type),
      }));

      // 2. Create Raw Edges
      const rawEdges = data.edges.map((e: any, i: number) => ({
        id: `e${i}`,
        source: e.source.toString(),
        target: e.target.toString(),
        animated: true,
        type: 'bezier', // Curvy lines prevent straight overlap
        style: { stroke: '#94a3b8', strokeWidth: 1.5 },
      }));

      // 3. Apply Top-Bottom Layout
      const layoutedNodes = getLayoutedElements(rawNodes, rawEdges);

      setNodes(layoutedNodes);
      setEdges(rawEdges);
    } catch (error) {
      console.error('Failed to fetch:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    generateArch();
  }, [generateArch]);

  return (
    <div className="page-content">
      <div className="page-toolbar">
        <button
          className="btn btn-primary"
          onClick={generateArch}
          disabled={loading}
        >
          {loading ? 'Generating...' : 'Generate Random Architecture'}
        </button>
        <span className="node-count">
          {nodes.length > 0 && `${nodes.length} nodes · ${edges.length} edges`}
        </span>
      </div>

      <div className="graph-container">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
          nodesDraggable={true} // Allow user to slightly adjust them if needed
        >
          <Background gap={20} size={1} color="#334155" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </div>
    </div>
  );
}