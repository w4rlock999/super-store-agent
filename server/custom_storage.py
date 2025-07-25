from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.storage.interface import Storage

class CustomStorage():

    def __init__(self):
        self.memories = []
        print(f"CustomStorage initialized")

    def save(self, role, value, metadata=None):
        self.memories.append({
            "role": role,
            "message": value
        })

        # print(f"Memory saved: {value}")

    def search(self, query, limit=10, score_threshold=0.5):
        # Implement your search logic here
        if len(self.memories) <= limit:
            return self.memories
        else:
            return self.memories[:limit]

    def reset(self):
        self.memories = []