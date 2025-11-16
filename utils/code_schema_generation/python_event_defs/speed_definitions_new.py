from dataclasses import dataclass
from typing import List, Optional
from utils_typing import Entity, Event, Relation, dataclass


@dataclass
class Event:
    pass

@dataclass
class Cure(Event):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Cure(mention='{self.mention}')"

@dataclass
class Spread(Event):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Spread(mention='{self.mention}')"

@dataclass
class Death(Event):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Death(mention='{self.mention}')"

@dataclass
class Infect(Event):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Infect(mention='{self.mention}')"

@dataclass
class Control(Event):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Control(mention='{self.mention}')"

@dataclass
class Prevent(Event):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Prevent(mention='{self.mention}')"

@dataclass
class Symptom(Event):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Symptom(mention='{self.mention}')"

