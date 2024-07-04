from math import cos, sin, pi


def apply_scaling(alg: str, decay: float, current_step: int, total_steps: int) -> float:
    if alg == "Flat":
        return decay

    ratio = float(current_step / total_steps)
    rad = ratio * pi / 2

    match alg:
        case "Cos":
            mod = cos(rad)
        case "Sin":
            mod = sin(rad)
        case "1 - Cos":
            mod = 1 - cos(rad)
        case "1 - Sin":
            mod = 1 - sin(rad)

    return decay * mod
