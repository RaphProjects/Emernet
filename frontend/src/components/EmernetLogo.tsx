export default function EmernetLogo() {
  const letters = [
    { offset: 0,   paths: ["M 60 0 L 0 0 L 0 100 L 60 100", "M 0 50 L 45 50"] }, // E
    { offset: 80,  paths: ["M 0 100 L 0 0 L 30 50 L 60 0 L 60 100"] },            // M
    { offset: 160, paths: ["M 60 0 L 0 0 L 0 100 L 60 100", "M 0 50 L 45 50"] }, // E
    { offset: 240, paths: ["M 0 100 L 0 0 L 60 0 L 60 50 L 0 50", "M 30 50 L 60 100"] }, // R
    { offset: 320, paths: ["M 0 100 L 0 0 L 60 100 L 60 0"] },                    // N
    { offset: 400, paths: ["M 60 0 L 0 0 L 0 100 L 60 100", "M 0 50 L 45 50"] }, // E
    { offset: 480, paths: ["M 0 0 L 60 0", "M 30 0 L 30 100"] },                  // T
  ];

  const nodes = [
    [0,0], [60,0], [0,100], [60,100], [0,50], [45,50], // E
    [80,100], [80,0], [110,50], [140,0], [140,100],    // M
    [160,0], [220,0], [160,100], [220,100], [160,50], [205,50], // E
    [240,100], [240,0], [300,0], [300,50], [240,50], [270,50], [300,100], // R
    [320,100], [320,0], [380,100], [380,0], // N
    [400,0], [460,0], [400,100], [460,100], [400,50], [445,50], // E
    [480,0], [540,0], [510,0], [510,100] // T
  ];

  // These lines bridge the gaps between letters
  const links = [
    [60,0, 80,0], [60,100, 80,100],
    [140,0, 160,0], [140,100, 160,100],
    [220,0, 240,0], [220,100, 240,100],
    [300,0, 320,0], [300,100, 320,100],
    [380,0, 400,0], [380,100, 400,100],
    [460,0, 480,0]
  ];

  return (
    <svg
        viewBox="-20 -20 580 140"
        style={{ 
            width: '100%', 
            maxWidth: '600px', 
            display: 'block', 
            margin: '0 auto'
        }}
        xmlns="http://www.w3.org/2000/svg"
        >
      <defs>
        <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        
        {/* Fixed gradient using userSpaceOnUse to prevent invisible lines */}
        <linearGradient 
          id="edgeGrad" 
          gradientUnits="userSpaceOnUse" 
          x1="0" y1="0" x2="540" y2="0"
        >
          <stop offset="0%" stopColor="#3b82f6" />
          <stop offset="50%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#3b82f6" />
        </linearGradient>
      </defs>

      {/* 1. Background Connections (Dashed) */}
      {links.map(([x1, y1, x2, y2], i) => (
        <line
          key={`link-${i}`}
          x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="#475569" strokeWidth="1.5" strokeDasharray="4 4"
        />
      ))}

      {/* 2. Main Letter Paths (Solid) */}
      <g>
        {letters.map((letter, index) => (
          <g key={`letter-${index}`} transform={`translate(${letter.offset}, 0)`}>
            {letter.paths.map((d, pIndex) => (
              <path
                key={`path-${pIndex}`}
                d={d}
                fill="none"
                stroke="url(#edgeGrad)"
                strokeWidth="5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          </g>
        ))}
      </g>

      {/* 3. Nodes (Vertices) */}
      {nodes.map(([x, y], i) => (
        <circle
          key={`node-${i}`}
          cx={x} cy={y} r="4"
          fill="#fff"
          stroke="#3b82f6"
          strokeWidth="2"
          filter="url(#nodeGlow)"
        />
      ))}
    </svg>
  );
}