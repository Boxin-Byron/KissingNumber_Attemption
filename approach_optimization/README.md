# Approach 1: Gradient Descent Optimization

This folder contains the implementation of the Gradient Descent approach for the Kissing Number problem.

## Structure
*   `src/`: Contains the core logic (optimizer, loss, utils).
*   `experiments/`: Scripts to run experiments.

## How to Run
1.  Install dependencies:
    ```bash
    pip install torch tqdm matplotlib numpy
    ```
2.  Run 2D experiment (Target N=6):
    ```bash
    python experiments/run_2d.py
    ```
3.  Run 3D experiment (Target N=12):
    ```bash
    python experiments/run_3d.py
    ```
