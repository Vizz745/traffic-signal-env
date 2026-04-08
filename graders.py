"""
graders.py — Root-level re-export of Traffic Signal Environment graders.

Allows OpenEnv validator to import graders directly without requiring installation.
Referenced in openenv.yaml as: graders:Task1Grader
"""

import sys
import os

# Ensure server/ is importable
_server_path = os.path.join(os.path.dirname(__file__), "server")
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)

try:
    # Installed package path (Docker / pip install)
    from traffic_signal_env.server.graders import Task1Grader, Task2Grader, Task3Grader
except ImportError:
    try:
        # Local repo path
        from server.graders import Task1Grader, Task2Grader, Task3Grader
    except ImportError:
        # Fallback
        from graders import Task1Grader, Task2Grader, Task3Grader  # type: ignore

__all__ = ["Task1Grader", "Task2Grader", "Task3Grader"]