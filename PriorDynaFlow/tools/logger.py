import os
import json
from datetime import datetime


class LocalLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.file = open(self.filename, "w")

    def log(self, data: dict):
        self.file.write(json.dumps(data) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()