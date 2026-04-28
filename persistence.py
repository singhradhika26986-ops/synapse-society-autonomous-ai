import json
from datetime import datetime, timezone
from pathlib import Path


class PersistenceManager:
    def __init__(self, root="data/runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, state, records=None, run_id=None):
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshots_path = run_dir / "snapshots.jsonl"
        entry = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "records": records or [],
        }
        with snapshots_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        (run_dir / "latest_state.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"run_id": run_id, "path": str(run_dir), "snapshot_count": self.snapshot_count(run_id)}

    def list_runs(self):
        runs = []
        for run_dir in sorted(self.root.glob("*"), reverse=True):
            if not run_dir.is_dir():
                continue
            latest_path = run_dir / "latest_state.json"
            latest = None
            if latest_path.exists():
                try:
                    latest = json.loads(latest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    latest = None
            runs.append(
                {
                    "run_id": run_dir.name,
                    "snapshot_count": self.snapshot_count(run_dir.name),
                    "latest_timestep": latest.get("state", {}).get("timestep") if latest else None,
                    "saved_at": latest.get("saved_at") if latest else None,
                }
            )
        return runs

    def load_latest(self, run_id):
        path = self.root / run_id / "latest_state.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def replay_frame(self, run_id, index):
        frames = self.load_frames(run_id)
        if not frames:
            return None
        index = max(0, min(index, len(frames) - 1))
        return {"index": index, "count": len(frames), "frame": frames[index]}

    def load_frames(self, run_id):
        path = self.root / run_id / "snapshots.jsonl"
        if not path.exists():
            return []
        frames = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    frames.append(json.loads(line))
        return frames

    def snapshot_count(self, run_id):
        path = self.root / run_id / "snapshots.jsonl"
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
