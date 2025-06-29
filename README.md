# 🛰️ Sensor-Thief Tracker

An intelligent multi-agent simulation in Python where autonomous "thieves" attempt to reach a goal while evading detection by strategically placed sensors. This project explores advanced AI techniques in path planning, real-time visibility analysis, and constraint-based system validation.

>🗓️ Created: May 2025 | 🎓 6th Semester, My Exploration of AI Pathfinding and Strategy

---

## Overview

This simulation models a stealth-based scenario where multiple thieves attempt to traverse a map to reach a designated goal. They must avoid detection from a network of sensors that:
- Have a fixed radius of vision.
- Are placed such that they form a partially connected communication graph.
- Can collectively "freeze" a thief if **K or more** sensors simultaneously detect them.

Thieves dynamically plan their path, avoid danger zones, and react to visibility threats in real time.

---

## Technologies Used

- **Python 3.x**
- **Pygame** – for GUI, animation, and input
- **NumPy** – for matrix calculations
- **Custom AI Algorithms** – pathfinding and constraint validation

---

## Key Features

### Constraint-Satisfying Map Generation

Before gameplay begins, the grid is constructed using constraint satisfaction logic:
- **Sensor Communication CSP**: Validates that at least a minimum number of sensor pairs are within communication radius.
- **Placement Constraints**: Ensures no overlapping entities, spatial separation, and edge-based thief placement.

### Realistic Sensor Vision

- **Line-of-Sight Calculation**: Uses Bresenham's algorithm to detect visibility between sensors and moving agents.
- **Occlusion Handling**: Walls block vision; sensors cannot see through them.

### Risk-Aware Thief Pathfinding

Thieves don’t simply follow the shortest path. Instead, each tile in the grid is assigned a **dynamic danger score** based on:

- Proximity to sensors
- Visibility exposure
- Terrain type (walls, sensors, etc.)

A custom A* algorithm then navigates thieves using this **weighted score map**, effectively prioritizing stealth over speed.

### Thief Freezing Logic

Each thief tracks how many sensors are observing them at each step. Once detected by **K or more sensors**, the thief becomes permanently frozen in place, and a red line is drawn to indicate the detection event.

---

## 🎨 UI & Gameplay

- **Pre-Game Setup Screen**: Input parameters like sensor radius, number of thieves, sensors, walls, and freeze threshold.
- **Animated Simulation**: Watch thieves move intelligently through the map, sensors scan zones, and the game react to every detection event.
- **Real-Time Feedback**: Debug outputs include visibility matrices, score maps, and thief path details.

---

## AI & Algorithm Summary

| Component                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **A* Pathfinding**       | Customized with score map to avoid danger areas                             |
| **Score Map**            | Weighted grid factoring in vision overlap, proximity, and obstacles         |
| **Visibility Matrix**    | Computed per-frame using line-of-sight between sensor and thief             |
| **Communication Matrix** | Ensures sensor network connectivity during initialization                   |
| **Constraint Validation**| CSP-style logic for ensuring a playable, fair, and valid map                |


