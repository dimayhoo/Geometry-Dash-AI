# Geometry Dash AI

A reinforcement learning project to train AI agents to play Geometry Dash using Python and C++.

## Project Overview

This project connects a C++ implementation of Geometry Dash (based on this adapted implementation: https://github.com/Open-GD/OpenGD/releases/tag/v0.1.0-beta) with Python-based reinforcement learning algorithms via pybind11. The goal is to create agents capable of mastering the precise timing and reflexes needed to complete Geometry Dash levels.

## Architecture

- **Game Engine**: C++ implementation of Geometry Dash 1.0 (OpenGD) for high-performance gameplay
- **Language Bridge**: pybind11 provides seamless interoperability between C++ game engine and Python ML code
- **Reinforcement Learning**: Custom environment and multiple agent implementations (DQN, A2C) in Python

## Key Features

The agent employs a strategic learning approach to master Geometry Dash by understanding gameplay patterns and level structures:

- **Level Structure Analysis**: Comprehensive parsing and analysis of level geometry, obstacles, and timing patterns
- **Multi-Modal Learning**: Incorporates level data, player state, and environmental context as model parameters
- **Predictive Action Planning**: Computes a vector of timely-based action and interacts based on it during the play

## Current Status

This project demonstrates the feasibility of connecting Geometry Dash with reinforcement learning but remains a work in progress. The main components are functional:

- ✅ C++ game implementation working
- ✅ Python bindings established
- ✅ RL environment interface complete
- ✅ Agent architecture implemented
- ✅ Initial training runs recorded
- ⏳ Full agent training to completion (pending)
- ⏳ Performance optimization by disabling rendering during training (planned)

## Technical Decisions

- **pybind11 over alternatives**: Chosen for its minimal overhead and seamless type conversion
- **Separate observation collection**: Game state data is collected separately from training to allow flexible experimentation and fast inference during the play
- **Multiple algorithms**: Both value-based (DQN) and policy-based (A2C) approaches implemented to compare performance
- **Custom OpenGD implementation**: Using an open-source GD clone enables full control over the game environment

## Limitations & Future Work

The agent has not yet achieved effective gameplay learning due to unresolved modeling challenges. The project requires further development in reinforcement learning methodology and performance optimization to reach its full potential. Key areas for improvement include:

- Training convergence and reward shaping
- Model architecture optimization for real-time gameplay
- Performance bottlenecks in the training pipeline
- Inspection of used RL algorithms

## Project Structure

- [`game`](game): C++ implementation of Geometry Dash using axmol engine
- [`cmake_example`](cmake_example): pybind11 bindings to connect C++ and Python
- [`python_`](python_): Reinforcement learning code (environment, agents, training)
  - `agent.py`: Implementation of RL algorithms
  - `env.py`: Game environment wrapper
  - `levelStructure.py`: Level analysis
