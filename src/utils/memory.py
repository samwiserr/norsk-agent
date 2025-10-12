# src/utils/memory.py
from collections import defaultdict, deque

class MemoryStore:
    def __init__(self, turns: int = 8):
        # per-session ring buffer to cap history
        self._mem = defaultdict(lambda: deque(maxlen=turns))

    def append(self, session_id: str | None, role: str, content: str):
        if not session_id:
            return
        self._mem[session_id].append({"role": role, "content": content})

    def get(self, session_id: str | None):
        if not session_id:
            return []
        return list(self._mem.get(session_id, []))

# single shared instance
memory = MemoryStore(turns=8)
