from dataclasses import dataclass
import json


@dataclass
class LabelingResult:
    query_name: str
    hint_set_int: int
    binary_rep: list[int]
    measured_time: float
    occurred_level: int
    is_opt: bool
    had_timeout: bool
    chosen_in_level: bool
    removed: bool
    seen_plan: bool
    hint_names: list[str]
    qep: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "query_name": self.query_name,
            "hint_set_int": self.hint_set_int,
            "time": self.measured_time,
            "level": self.occurred_level,
            "opt": self.is_opt,
            "timeout": self.had_timeout,
            "chosen": self.chosen_in_level,
            "removed": self.removed,
            "seen_plan": self.seen_plan,
            "qep": json.dumps(self.qep)
        }
        result.update({name: self.binary_rep[i] for i, name in enumerate(self.hint_names)})
        return result
