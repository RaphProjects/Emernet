export default function EmernetLogo() {
  const letters = [
    { offset: 0,   paths: ["M 60 0 L 0 0 L 0 100 L 60 100", "M 0 50 L 45 50"] },
    { offset: 80,  paths: ["M 0 100 L 0 0 L 30 50 L 60 0 L 60 100"] },
    { offset: 160, paths: ["M 60 0 L 0 0 L 0 100 L 60 100", "M 0 50 L 45 50"] },
    { offset: 240, paths: ["M 0 100 L 0 0 L 60 0 L 60 50 L 0 50", "M 30 50 L 60 100"] },
    { offset: 320, paths: ["M 0 100 L 0 0 L 60 100 L 60 0"] },
    { offset: 400, paths: ["M 60 0 L 0 0 L 0 100 L 60 100", "M 0 50 L 45 50"] },
    { offset: 480, paths: ["M 0 0 L 60 0", "M 30 0 L 30 100"] },
  ];

  const nodes = [
    [0,0], [60,0], [0,100], [60,100], [0,50], [45,50],
    [80,100], [80,0], [110,50], [140,0], [140,100],
    [160,0], [220,0], [160,100], [220,100], [160,50], [205,50],
    [240,100], [240,0], [300,0], [300,50], [240,50], [270,50], [300,100],
    [320,100], [320,0], [380,100], [380,0],
    [400,0], [460,0], [400,100], [460,100], [400,50], [445,50],
    [480,0], [540,0], [510,0], [510,100]
  ];

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
        maxWidth: '520px',
        minHeight: '80px',
        flexShrink: 0,
        display: 'block',
        margin: '0 auto',
      }}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="logoGrad" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="540" y2="0">
          <stop offset="0%" stopColor="#ff7a18">
            <animate attributeName="stopColor" values="#ff7a18;#fff23d;#ffb000;#ff7a18" dur="6s" repeatCount="indefinite" />
          </stop>
          <stop offset="50%" stopColor="#fff23d">
            <animate attributeName="stopColor" values="#fff23d;#ffb000;#ff7a18;#fff23d" dur="6s" repeatCount="indefinite" />
          </stop>
          <stop offset="100%" stopColor="#ff7a18">
            <animate attributeName="stopColor" values="#ff7a18;#fff23d;#ffb000;#ff7a18" dur="6s" repeatCount="indefinite" />
          </stop>
        </linearGradient>

        <filter id="logoGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {links.map(([x1, y1, x2, y2], i) => (
        <line
          key={`link-${i}`}
          x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="rgba(255,122,24,0.24)"
          strokeWidth="1.5"
          strokeDasharray="4 4"
        >
          <animate attributeName="stroke-opacity" values="0.15;0.4;0.15" dur={`${3 + i * 0.3}s`} repeatCount="indefinite" />
        </line>
      ))}

      <g filter="url(#logoGlow)">
        {letters.map((letter, index) => (
          <g key={`letter-${index}`} transform={`translate(${letter.offset}, 0)`}>
            {letter.paths.map((d, pIndex) => (
              <path
                key={`path-${pIndex}`}
                d={d}
                fill="none"
                stroke="url(#logoGrad)"
                strokeWidth="5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          </g>
        ))}
      </g>

      {nodes.map(([x, y], i) => (
        <circle
          key={`node-${i}`}
          cx={x} cy={y} r="3.5"
          fill="#050403"
          stroke="url(#logoGrad)"
          strokeWidth="2"
          filter="url(#nodeGlow)"
        >
          <animate attributeName="r" values="3.5;5;3.5" dur={`${2 + (i % 5) * 0.4}s`} repeatCount="indefinite" />
          <animate attributeName="stroke-width" values="2;1;2" dur={`${2 + (i % 5) * 0.4}s`} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}
