import time


class Timer:

    def __init__(self):
        self._start_time = 0

    def start(self):
        self._start_time = time.perf_counter()
        return self

    def stop(self, round_number: int = 0) -> float:
        result = time.perf_counter() - self._start_time
        if round_number > 0:
            result = round(result, round_number)
        return result

    def fps(self) -> int:
        return int(1 / self.stop())

    def fps_str(self) -> str:
        return str(self.fps())
