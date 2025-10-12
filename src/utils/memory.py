from collections import defaultdict, deque

class MemoryStore:
    def __init__(self, turns=6):
        self._mem = defaultdict(lambda: deque(maxlen=turns))

    def append(self, session_id: str, role: str, content: str):
        self._mem[session_id].append({"role": role, "content": content})

    def get(self, session_id: str):
        return list(self._mem[session_id])

memory = MemoryStore(turns=8)
