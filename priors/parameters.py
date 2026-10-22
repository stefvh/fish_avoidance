from enum import Enum

class Dummy(Enum):
    FISH = "Fish"
    PREDATOR = "Predator"

class SlowSpeed(Enum):
    LOW = 1.2
    HIGH = 1.6

class FastSpeed(Enum):
    LOW = 1.6
    HIGH = 1.9

class TransitionRate(Enum):
    LOW = 0.5
    MID = 1.0
    HIGH = 10.0

class TransitionDistance(Enum):
    SMALL = 15.0
    LARGE = 30.0

class StartingDistance(Enum):
    DEFAULT = 40.0

PARAMETER_SET_1_FISH = []
PARAMETER_SET_1_PREDATOR = []

PARAMETER_SET_2_FISH = []

for slow_speed in [SlowSpeed.LOW, SlowSpeed.HIGH]:
    for transition_distance in TransitionDistance:
        PARAMETER_SET_1_FISH.append({
            "dummy": Dummy.FISH.value,
            "slow_speed": slow_speed.value,
            "fast_speed": FastSpeed.HIGH.value,
            "transition_rate": TransitionRate.HIGH.value,
            "transition_distance": transition_distance.value,
            "starting_distance": StartingDistance.DEFAULT.value,
        })
        PARAMETER_SET_1_PREDATOR.append({
            "dummy": Dummy.PREDATOR.value,
            "slow_speed": slow_speed.value,
            "fast_speed": FastSpeed.HIGH.value,
            "transition_rate": TransitionRate.HIGH.value,
            "transition_distance": transition_distance.value,
            "starting_distance": StartingDistance.DEFAULT.value,
        })

for slow_speed in [SlowSpeed.LOW, SlowSpeed.HIGH]:
    for fast_speed in FastSpeed:
        for transition_rate in TransitionRate:
            for transition_distance in TransitionDistance:
                PARAMETER_SET_2_FISH.append({
                    "dummy": Dummy.FISH.value,
                    "slow_speed": slow_speed.value,
                    "fast_speed": fast_speed.value,
                    "transition_rate": transition_rate.value,
                    "transition_distance": transition_distance.value,
                    "starting_distance": StartingDistance.DEFAULT.value,
                })

def get_parameter_set(id):
    if id == "FISH_1":
        return PARAMETER_SET_1_FISH
    elif id == "PREDATOR_1":
        return PARAMETER_SET_1_PREDATOR
    elif id == "FISH_2":
        return PARAMETER_SET_2_FISH
    else:
        raise ValueError("Invalid parameter set id")
