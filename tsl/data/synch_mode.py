from enum import Enum


class SynchMode(Enum):
    WINDOW = 'window'
    HORIZON = 'horizon'
    STATIC = 'static'
    PAST = 'past'
    HISTORY = 'history'


# Aliases
WINDOW = SynchMode.WINDOW
HORIZON = SynchMode.HORIZON
STATIC = SynchMode.STATIC
PAST = SynchMode.PAST
HISTORY = SynchMode.HISTORY
