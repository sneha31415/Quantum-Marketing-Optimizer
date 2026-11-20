"""A single persistent thread that owns Qiskit.

Why this exists
---------------
Streamlit executes every script rerun in a *new* thread. Qiskit 2.x contains Rust extension code, and if Qiskit is first
imported in a thread that later exits, using it from another thread can crash the whole process with a segmentation fault
(reproduced with plain Python threads, see README "Design notes"). The fix is to import and use Qiskit ONLY inside one
long-lived thread. The UI thread submits jobs by *function name* and receives plain data (numpy / pandas / matplotlib
figures), so it never touches Qiskit objects.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future


class QuantumWorker:
    def __init__(self):
        self._q: "queue.Queue" = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qiskit-worker", daemon=True)
        self._thread.start()

    def _loop(self):
        import importlib
        core = importlib.import_module("qmo.core")      # first (and only) Qiskit import happens in THIS thread
        self._ready.set()
        while True:
            item = self._q.get()
            if item is None:
                return
            fut, name, args, kwargs = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(getattr(core, name)(*args, **kwargs))
            except BaseException as e:   # noqa: BLE001 - propagate any failure to the caller
                fut.set_exception(e)

    def submit(self, name: str, *args, **kwargs) -> Future:
        fut: Future = Future()
        self._q.put((fut, name, args, kwargs))
        return fut

    def call(self, name: str, *args, **kwargs):
        return self.submit(name, *args, **kwargs).result()

    def versions(self) -> dict:
        import importlib.metadata as md
        return {"qiskit": md.version("qiskit"), "qiskit-aer": md.version("qiskit-aer")}
