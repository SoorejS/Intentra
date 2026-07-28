"""
Intentra Async Job Manager & SSE Event Hub.

Handles background generation tasks, status tracking, and event streaming for real-time progress updates.
"""

import asyncio
import uuid
import time
import json
from typing import Dict, Any, AsyncGenerator

class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self) -> str:
        """Create a new job and return its unique UUID."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "id": job_id,
            "status": "processing",
            "progress": 0,
            "events": [],
            "listeners": set(),
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        self._cleanup_old_jobs()
        return job_id

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        """Get the current job status and metadata."""
        return self._jobs.get(job_id)

    async def emit_event(self, job_id: str, event_type: str, data: Dict[str, Any]):
        """Emit an SSE event to all current and future listeners of a job."""
        job = self._jobs.get(job_id)
        if not job:
            return

        event_payload = {
            "event": event_type,
            "data": data,
            "timestamp": time.time()
        }

        job["events"].append(event_payload)
        job["updated_at"] = time.time()

        if event_type == "status":
            job["progress"] = data.get("progress", job["progress"])
        elif event_type == "schema":
            job["progress"] = data.get("progress", job["progress"])
        elif event_type == "batch":
            job["progress"] = data.get("progress", job["progress"])
        elif event_type == "complete":
            job["status"] = "completed"
            job["progress"] = 100
            job["result"] = data
        elif event_type in ("error", "job_error"):
            job["status"] = "failed"
            job["error"] = data.get("detail", "Job failed")

        # Notify active listener queues
        for queue in list(job["listeners"]):
            await queue.put(event_payload)

    async def stream_events(self, job_id: str) -> AsyncGenerator[str, None]:
        """Async generator yielding SSE formatted strings for a given job."""
        job = self._jobs.get(job_id)
        if not job:
            yield f"event: error\ndata: {json.dumps({'detail': 'Job not found'})}\n\n"
            return

        queue = asyncio.Queue()
        job["listeners"].add(queue)

        try:
            # Replay historical events first
            for event in list(job["events"]):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

            # If job is already complete or failed, we are done
            if job["status"] in ("completed", "failed"):
                return

            # Stream live events as they arrive
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                    if event["event"] in ("complete", "error", "job_error"):
                        break
                except asyncio.TimeoutError:
                    # Keep-alive ping
                    yield ": ping\n\n"
                    if job["status"] in ("completed", "failed"):
                        break
        finally:
            job["listeners"].discard(queue)

    def _cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Remove jobs older than max_age_seconds."""
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items()
            if now - j["created_at"] > max_age_seconds
        ]
        for jid in expired:
            del self._jobs[jid]

# Global singleton instance
job_manager = JobManager()
