from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse
from typing import Optional
import uuid

try:
    from models import TrafficSignalAction, TrafficSignalObservation
    from server.traffic_signal_env_environment import TrafficSignalEnvironment
    from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3
except ImportError:
    from ..models import TrafficSignalAction, TrafficSignalObservation
    from .traffic_signal_env_environment import TrafficSignalEnvironment
    from tasks import EpisodeRecord, grade_task1, grade_task2, grade_task3

app = FastAPI(title="Traffic Signal Control Environment")

sessions = {}


def _make_record(session_id):
    return sessions[session_id]["record"]

def _compute_score(session_id) -> float:
    s = sessions[session_id]
    record = s["record"]
    task_id = record.task_id
    response_steps = s["response_steps"]
    cleared = s["cleared"]
    total_emg = s["total_emg"]
    if task_id == "task1":
        return grade_task1(record)
    elif task_id == "task2":
        return grade_task2(record, response_steps, cleared, total_emg)
    else:
        return grade_task3(record, response_steps, cleared, total_emg)


@app.post("/reset")
def reset(task_id: str = "task1", x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id or str(uuid.uuid4())
    env = TrafficSignalEnvironment()
    obs = env.reset(task_id=task_id)
    sessions[session_id] = {
        "env": env,
        "record": EpisodeRecord(task_id=task_id),
        "response_steps": [],
        "cleared": 0,
        "total_emg": 0,
        "emg_active": False,
        "current_emg_steps": 0,
    }
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "session_id": session_id,
    }


@app.post("/step")
def step(payload: dict, x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id
    if not session_id or session_id not in sessions:
        if sessions:
            session_id = list(sessions.keys())[-1]
        else:
            return {"error": "No active session. Call /reset first."}

    s = sessions[session_id]
    env = s["env"]
    record = s["record"]

    action_data = payload.get("action", payload)
    action = TrafficSignalAction(**action_data)
    obs = env.step(action)

    # Update record
    record.update(obs)

    # Track emergency events
    if obs.emergency_direction:
        if not s["emg_active"]:
            s["emg_active"] = True
            s["total_emg"] += 1
            s["current_emg_steps"] = 0
        s["current_emg_steps"] += 1
    elif s["emg_active"]:
        s["emg_active"] = False
        s["cleared"] += 1
        s["response_steps"].append(s["current_emg_steps"])

    response = {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "session_id": session_id,
    }

    # Include score in final step response
    if obs.done:
        response["score"] = _compute_score(session_id)

    return response


@app.get("/state")
def state(x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id
    if not session_id or session_id not in sessions:
        if sessions:
            session_id = list(sessions.keys())[-1]
        else:
            return {"episode_id": "", "step_count": 0}
    return sessions[session_id]["env"].state.model_dump()


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Traffic Signal Control Environment</title>
  <style>
    body { font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }
    h1 { color: #4fc3f7; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
    .card { background: #2a2a2a; border-radius: 8px; padding: 16px; }
    .card h3 { margin: 0 0 12px 0; color: #81c784; }
    .queue-bar { height: 20px; background: #4fc3f7; margin: 4px 0; border-radius: 3px; transition: width 0.3s; }
    .emergency { color: #ef5350; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 50% { opacity: 0.3; } }
    .phase-ns { color: #66bb6a; }
    .phase-ew { color: #ffa726; }
    button {
      background: #4fc3f7; color: #000; border: none;
      padding: 10px 20px; border-radius: 6px; cursor: pointer;
      font-size: 14px; margin: 4px;
    }
    button:hover { background: #81d4fa; }
    button.active { background: #66bb6a; }
    .log { background: #111; padding: 10px; border-radius: 6px;
           height: 200px; overflow-y: auto; font-size: 12px; }
    .log div { border-bottom: 1px solid #222; padding: 3px 0; }
    .reward-pos { color: #66bb6a; }
    .reward-neg { color: #ef5350; }
    select { background: #2a2a2a; color: #e0e0e0; border: 1px solid #444;
             padding: 6px; border-radius: 4px; }
    .stat { display: flex; justify-content: space-between; margin: 4px 0; }
    .stat span:last-child { color: #4fc3f7; }
  </style>
</head>
<body>
  <h1>🚦 Traffic Signal Control Environment</h1>
  <div style="margin-bottom:16px">
    <label>Task:
      <select id="taskSelect">
        <option value="task1">Task 1 — Easy (Rush hour, no emergencies)</option>
        <option value="task2">Task 2 — Medium (Balanced + emergencies)</option>
        <option value="task3">Task 3 — Hard (Rush hour + emergencies + fairness)</option>
      </select>
    </label>
    <button onclick="resetEnv()">Reset</button>
    <button onclick="toggleAuto()" id="autoBtn">▶ Auto Step</button>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Intersection State</h3>
      <div id="phase" style="font-size:18px;margin-bottom:12px"></div>
      <div id="emergency"></div>
      <br>
      <div>North: <span id="nq">0</span>
        <div class="queue-bar" id="nb" style="width:0%"></div>
      </div>
      <div>South: <span id="sq">0</span>
        <div class="queue-bar" id="sb" style="width:0%"></div>
      </div>
      <div>East: <span id="eq">0</span>
        <div class="queue-bar" id="eb" style="width:0%"></div>
      </div>
      <div>West: <span id="wq">0</span>
        <div class="queue-bar" id="wb" style="width:0%"></div>
      </div>
    </div>
    <div class="card">
      <h3>Episode Stats</h3>
      <div class="stat"><span>Step</span><span id="step">0</span></div>
      <div class="stat"><span>Throughput</span><span id="tp">0</span></div>
      <div class="stat"><span>Total Wait</span><span id="tw">0</span></div>
      <div class="stat"><span>Last Reward</span><span id="rew">0</span></div>
      <div class="stat"><span>Score</span><span id="score">—</span></div>
      <br>
      <h3>Manual Control</h3>
      <button id="nsBtn" onclick="manualStep('NS_GREEN')">NS_GREEN</button>
      <button id="ewBtn" onclick="manualStep('EW_GREEN')">EW_GREEN</button>
      <br><br>
      <div id="hint" style="color:#ffb74d;font-size:13px"></div>
    </div>
  </div>
  <div class="card" style="margin-top:20px">
    <h3>Step Log</h3>
    <div class="log" id="log"></div>
  </div>
<script>
  let autoInterval = null;
  let isAuto = false;
  let currentTask = 'task1';
  let done = false;
  let sessionId = null;

  async function resetEnv() {
    currentTask = document.getElementById('taskSelect').value;
    done = false;
    document.getElementById('score').textContent = '—';
    const r = await fetch('/reset?task_id=' + currentTask, {method:'POST'});
    const data = await r.json();
    sessionId = data.session_id;
    updateUI(data, null);
    document.getElementById('log').innerHTML = '';
    addLog('Environment reset — ' + currentTask, 0);
  }

  async function manualStep(phase) {
    if (done) { addLog('Episode done — press Reset', 0); return; }
    const headers = {'Content-Type':'application/json'};
    if (sessionId) headers['x-session-id'] = sessionId;
    const r = await fetch('/step', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({action: {phase: phase, task_id: currentTask}})
    });
    const data = await r.json();
    updateUI(data, phase);
    if (data.done) {
      done = true;
      if (data.score !== undefined) {
        document.getElementById('score').textContent = data.score.toFixed(4);
      }
      addLog('=== Episode Complete — Score: ' + (data.score || '?') + ' ===', 0);
    }
  }

  function updateUI(data, phase) {
    const obs = data.observation;
    const maxQ = 20;
    const phaseEl = document.getElementById('phase');
    phaseEl.innerHTML = obs.current_phase === 'NS_GREEN'
      ? '<span class="phase-ns">● NS_GREEN</span> (North+South green)'
      : '<span class="phase-ew">● EW_GREEN</span> (East+West green)';
    const emgEl = document.getElementById('emergency');
    emgEl.innerHTML = obs.emergency_direction
      ? '<span class="emergency">🚨 EMERGENCY: ' + obs.emergency_direction + ' (' + obs.emergency_steps_remaining + ' steps)</span>'
      : '<span style="color:#666">No emergency</span>';
    const setQ = (id, bar, val) => {
      document.getElementById(id).textContent = val;
      document.getElementById(bar).style.width = Math.min(100, val/maxQ*100) + '%';
    };
    setQ('nq','nb', obs.north_queue);
    setQ('sq','sb', obs.south_queue);
    setQ('eq','eb', obs.east_queue);
    setQ('wq','wb', obs.west_queue);
    document.getElementById('step').textContent = obs.step;
    document.getElementById('tp').textContent = obs.throughput;
    document.getElementById('tw').textContent = obs.total_wait_time;
    const rew = data.reward || 0;
    document.getElementById('rew').innerHTML =
      '<span class="' + (rew >= 0 ? 'reward-pos':'reward-neg') + '">' + rew.toFixed(4) + '</span>';
    document.getElementById('hint').textContent = obs.hint;
    document.getElementById('nsBtn').className = obs.current_phase === 'NS_GREEN' ? 'active' : '';
    document.getElementById('ewBtn').className = obs.current_phase === 'EW_GREEN' ? 'active' : '';
    if (phase) addLog(phase + ' → reward: ' + rew.toFixed(4) + ' | N:' + obs.north_queue + ' S:' + obs.south_queue + ' E:' + obs.east_queue + ' W:' + obs.west_queue, rew);
  }

  function addLog(msg, rew) {
    const log = document.getElementById('log');
    const div = document.createElement('div');
    div.className = rew > 0 ? 'reward-pos' : rew < 0 ? 'reward-neg' : '';
    div.textContent = msg;
    log.prepend(div);
  }

  function toggleAuto() {
    isAuto = !isAuto;
    document.getElementById('autoBtn').textContent = isAuto ? '⏸ Stop Auto' : '▶ Auto Step';
    if (isAuto) {
      autoInterval = setInterval(() => {
        if (done) { toggleAuto(); return; }
        const ns = parseInt(document.getElementById('nq').textContent) +
                   parseInt(document.getElementById('sq').textContent);
        const ew = parseInt(document.getElementById('eq').textContent) +
                   parseInt(document.getElementById('wq').textContent);
        manualStep(ns >= ew ? 'NS_GREEN' : 'EW_GREEN');
      }, 600);
    } else {
      clearInterval(autoInterval);
    }
  }

  resetEnv();
</script>
</body>
</html>
"""


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()