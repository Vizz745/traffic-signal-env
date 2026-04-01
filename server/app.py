from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from models import TrafficSignalAction, TrafficSignalObservation
    from server.traffic_signal_env_environment import TrafficSignalEnvironment
except ImportError:
    from ..models import TrafficSignalAction, TrafficSignalObservation
    from .traffic_signal_env_environment import TrafficSignalEnvironment

app = FastAPI(title="Traffic Signal Control Environment")

# Single shared instance — persists state across requests
env = TrafficSignalEnvironment()


@app.post("/reset")
def reset(task_id: str = "task1"):
    obs = env.reset(task_id=task_id)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}


@app.post("/step")
def step(payload: dict):
    action_data = payload.get("action", payload)
    action = TrafficSignalAction(**action_data)
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
def state():
    return env.state.model_dump()


@app.get("/health")
def health():
    return {"status": "ok"}

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()   