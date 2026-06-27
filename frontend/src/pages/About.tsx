const sections = [
  {
    title: 'The Core Idea',
    accent: 'var(--theme-primary)',
    content: (
      <>
        <p>
          Neural Architecture Search (NAS) traditionally requires training candidate architectures on specific benchmark datasets. This is incredibly expensive, task-biased, and arbitrary.
        </p>
        <p>
          <strong>Emernet</strong> asks: <em>Can we evaluate the quality of a neural architecture without using any real-world data at all?</em> We do this through a <strong>Mutual Imitation Tournament</strong>. Two architectures are initialized randomly, producing random mathematical functions. They then try to learn to imitate each other's functions.
        </p>
        <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', marginTop: '16px' }}>
          <a href="https://github.com/RaphProjects/Emernet" target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href="https://youtu.be/8x_sJtFG7F4" target="_blank" rel="noopener noreferrer">Presentation video</a>
        </div>
      </>
    ),
  },
  {
    title: 'Current Challenges',
    accent: 'var(--theme-accent)',
    content: (
      <ul>
        <li><strong>Normalization Fragility:</strong> Scores are Z-normalized against pre-computed global baselines, so module math changes require recomputing global averages and deviations.</li>
        <li><strong>Validation Bottleneck:</strong> Proving correlation between arena performance and real-world performance requires a massive, statistically significant sample.</li>
      </ul>
    ),
  },
  {
    title: 'Future Directions',
    accent: '#66ff99',
    content: (
      <ul>
        <li><strong>Interactive Architecture Builder:</strong> Manually edit, add, or remove modules directly in the Architecture Viewer.</li>
        <li><strong>Intelligent Search:</strong> Use RL or evolutionary algorithms to mutate tournament winners and navigate the architecture space.</li>
      </ul>
    ),
  },
  {
    title: 'Use of AI',
    accent: '#ff4d2e',
    content: (
      <p>
        The frontend of this application was primarily generated with AI assistance. The author's expertise lies in the Python backend, neural architecture design, and research concepts. All core logic is original work.
      </p>
    ),
  },
];

export default function About() {
  return (
    <div style={{ flex: 1, overflowY: 'auto', height: '100%' }}>
      <div style={{ maxWidth: '860px', margin: '0 auto', padding: '40px 20px' }}>
        <h1 style={{ color: 'var(--text-primary)', fontSize: '32px', marginBottom: '8px' }}>About Emernet</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '16px', marginBottom: '32px' }}>
          Zero-Data Neural Architecture Search via Mutual Imitation
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {sections.map(section => (
            <section
              key={section.title}
              className="glass-panel"
              style={{
                padding: '24px',
                borderRadius: '8px',
                borderLeft: `4px solid ${section.accent}`,
                color: 'var(--text-secondary)',
                lineHeight: '1.6',
              }}
            >
              <h2 style={{ color: 'var(--text-primary)', marginTop: 0, marginBottom: '8px', fontSize: '20px' }}>
                {section.title}
              </h2>
              <div className="about-copy">
                {section.content}
              </div>
            </section>
          ))}
        </div>
        <div style={{ height: '40px' }} />
      </div>
    </div>
  );
}
