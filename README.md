# Emernet

Zero-data neural architecture search through mutual imitation tournaments.

## Overview

Emernet evaluates neural network architectures without using real datasets. It works by measuring how well randomly initialized architectures can imitate each other in a round-robin tournament, producing a task-agnostic estimate of architectural quality.

The primary finding is that this zero-data signal, particularly the **learnability** metric, correlates with downstream performance on real benchmarks, suggesting that architecture quality can be estimated without training on real data.

## How It Works

Two architectures are initialized with random weights. Each one produces a target function from its random initialization, and the other tries to imitate it. The outcome of this pairwise comparison yields three metrics:

| Metric           | Description                                                                   |
| ---------------- | ------------------------------------------------------------------------------|
| **Learnability** | How well an architecture can imitate others.                                  |
| **Simplicity**   | How easily others can imitate it. Showed no advantage so it is no longer used |
| **Occam score**  | Learnability weighted by speed.                                               |

## Tech Stack

**Frontend** — React, TypeScript, Vite, React Flow, Recharts

**Backend** — FastAPI, PyTorch, NetworkX, scikit-learn

**Communication** — REST API, WebSocket (live tournament updates)

## Features

- Random architecture generation with DAG visualization
- Fight viewer for pairwise architecture comparison
- Tournament viewer for round-robin evaluation
- Upload `.pkl` architectures from your machine
- Download architectures as `.pkl` files

## Project Structure

```
frontend/
  src/
    pages/
    components/

backend/
  server.py
  fight_viz.py

graph/
  architecture.py
  executor.py
  generator.py

modules/
  ...

tournament/
  arena.py
```

## Getting Started

### Backend

```bash
uvicorn server:app --reload
```

### Frontend

```bash
npm install
npm run dev
```

### Environment Variables

The frontend expects the following variables:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_WS_BASE_URL=ws://127.0.0.1:8000
```

For production, point them to your deployed backend:

```env
VITE_API_BASE_URL=https://your-space.hf.space
VITE_WS_BASE_URL=wss://your-space.hf.space
```
