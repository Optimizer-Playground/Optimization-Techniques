from dataclasses import dataclass


@dataclass
class HintSetStats:
    used: int = 0
    better: int = 0

    def update(self, was_better: bool):
        self.used += 1
        if was_better:
            self.better += 1


def get_success_ratio(stats: HintSetStats, smoothing: float = 1.0) -> float:
    return (stats.better + smoothing) / (stats.used + 2 * smoothing)


def get_active_hints(hint_int: int) -> set[int]:
    return {1 << i for i in range(hint_int.bit_length()) if hint_int & (1 << i)}


def get_neighbors(
        hs_int: int,
        experience: dict[int, HintSetStats],
        hint_restrictions: set[int],
        default_score: float = 0.5
) -> list[int]:

    active = get_active_hints(hs_int)
    candidates = [
        (bit, hs_int - bit)
        for bit in active
        if bit not in hint_restrictions
    ]

    sorted_neighbors = sorted(
        candidates,
        key=lambda pair: 1.0 - get_success_ratio(experience.get(pair[1]), smoothing=1.0)
        if pair[1] in experience else 1.0 - default_score
    )

    return [neighbor for _, neighbor in sorted_neighbors]
