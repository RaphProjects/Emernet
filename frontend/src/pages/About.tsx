export default function About() {
  return (
    <div style={{ flex: 1, overflowY: 'auto', height: '100%' }}>
      <div style={{ maxWidth: '800px', margin: '0 auto', padding: '40px 20px' }}>
        <h1 style={{ color: '#f1f5f9', fontSize: '32px', marginBottom: '8px' }}>About Emernet</h1>
        <p style={{ color: '#94a3b8', fontSize: '16px', marginBottom: '32px' }}>
          Zero-Data Neural Architecture Search via Mutual Imitation
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

          <div style={{ backgroundColor: '#1e293b', padding: '24px', borderRadius: '8px', borderLeft: '4px solid #3b82f6' }}>
            <h2 style={{ color: 'white', marginTop: 0, marginBottom: '8px', fontSize: '20px' }}>The Core Idea</h2>
            <p style={{ color: '#cbd5e1', lineHeight: '1.6' }}>
              Neural Architecture Search (NAS) traditionally requires training candidate architectures on specific benchmark datasets. This is incredibly expensive, task-biased, and arbitrary.
            </p>
            <p style={{ color: '#cbd5e1', lineHeight: '1.6', marginBottom: 0 }}>
              <strong>Emernet</strong> asks: <em>Can we evaluate the quality of a neural architecture without using any real-world data at all?</em> We do this through a <strong>Mutual Imitation Tournament</strong>. Two architectures are initialized randomly, producing random mathematical functions. They then try to learn to imitate each other's functions. The quality of this imitation reveals the architecture's inherent inductive bias and structural quality.
            </p>
            <div style={{ marginTop: '16px' }}>
              <a
                href="https://github.com/RaphProjects/Emernet"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: '#60a5fa', textDecoration: 'none', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                View the project on GitHub
              </a>
            </div>
            <div style={{ marginTop: '16px' }}>
              <a
                href="https://youtu.be/8x_sJtFG7F4"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: '#60a5fa', textDecoration: 'none', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                Presentation video
              </a>
            </div>
          </div>

          <div style={{ backgroundColor: '#1e293b', padding: '24px', borderRadius: '8px', borderLeft: '4px solid #f59e0b' }}>
            <h2 style={{ color: 'white', marginTop: 0, fontSize: '20px', marginBottom: '8px' }}>Current Challenges</h2>
            <ul style={{ color: '#cbd5e1', lineHeight: '1.6', margin: 0, paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <li>
                <strong>Normalization Fragility:</strong> Scores are Z-normalized against pre-computed global baselines. Any time a module's math is edited or added, the global averages and standard deviations for <em>learnability</em> and <em>speed</em> must be entirely recomputed over hundreds of fights.
              </li>
              <li>
                <strong>Validation Bottleneck:</strong> Proving the correlation between arena performance (zero-data) and real-world dataset performance requires a massive sample size. The inherently noisy nature of random weight initialization combined with the high compute time of training makes gathering statistically significant proof very challenging.
              </li>
            </ul>
          </div>


          <div style={{ backgroundColor: '#1e293b', padding: '24px', borderRadius: '8px', borderLeft: '4px solid #10b981' }}>
            <h2 style={{ color: 'white', marginTop: 0, fontSize: '20px', marginBottom: '8px' }}>Future Directions</h2>
            <ul style={{ color: '#cbd5e1', lineHeight: '1.6', margin: 0, paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <li>
                <strong>Interactive Architecture Builder:</strong> Allowing users to manually edit, add, or remove modules directly within the Architecture Viewer interface, rather than relying strictly on random generation.
              </li>
              <li>
                <strong>Intelligent Search:</strong> Moving away from pure random architecture generation. Implementing Reinforcement Learning (RL) or Evolutionary Algorithms to mutate tournament winners and intelligently navigate the architecture search space.
              </li>
            </ul>
          </div>


          <div style={{ backgroundColor: '#1e293b', padding: '24px', borderRadius: '8px', borderLeft: '4px solid #8b5cf6' }}>
            <h2 style={{ color: 'white', marginTop: 0, fontSize: '20px',marginBottom: '8px' }}>Use of AI</h2>
            <p style={{ color: '#cbd5e1', lineHeight: '1.6', margin: 0 }}>
              The frontend of this application (React / TypeScript) was primarily generated with the assistance of AI. The author's expertise lies in the Python backend, the neural architecture design, and the research concepts, not in web development. All core logic (architecture generation, tournament scoring, the mutual imitation framework) is original work.
            </p>
          </div>

        </div>
        <div style={{ height: '40px' }} />
      </div>
    </div>
  );
}