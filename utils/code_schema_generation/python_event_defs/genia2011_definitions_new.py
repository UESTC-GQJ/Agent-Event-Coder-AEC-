from dataclasses import dataclass
from typing import List, Optional
from utils_typing import Entity, Event, Relation, dataclass


@dataclass
class Event:
    pass

@dataclass
class Positive_regulation(Event):
    def __init__(self, mention: Optional[str] = None, theme: Optional[List] = None, csite: Optional[List] = None, site: Optional[List] = None, cause: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.theme = theme if theme is not None else []
        self.csite = csite if csite is not None else []
        self.site = site if site is not None else []
        self.cause = cause if cause is not None else []
    def __repr__(self):
        return f"Positive_regulation(mention='{self.mention}', theme={self.theme}, csite={self.csite}, site={self.site}, cause={self.cause})"

@dataclass
class Phosphorylation(Event):
    def __init__(self, mention: Optional[str] = None, theme: Optional[List] = None, site: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.theme = theme if theme is not None else []
        self.site = site if site is not None else []
    def __repr__(self):
        return f"Phosphorylation(mention='{self.mention}', theme={self.theme}, site={self.site})"

@dataclass
class Gene_expression(Event):
    def __init__(self, mention: Optional[str] = None, theme: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.theme = theme if theme is not None else []
    def __repr__(self):
        return f"Gene_expression(mention='{self.mention}', theme={self.theme})"

@dataclass
class Regulation(Event):
    def __init__(self, mention: Optional[str] = None, theme: Optional[List] = None, csite: Optional[List] = None, site: Optional[List] = None, cause: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.theme = theme if theme is not None else []
        self.csite = csite if csite is not None else []
        self.site = site if site is not None else []
        self.cause = cause if cause is not None else []
    def __repr__(self):
        return f"Regulation(mention='{self.mention}', theme={self.theme}, csite={self.csite}, site={self.site}, cause={self.cause})"

@dataclass
class Binding(Event):
    def __init__(self, mention: Optional[str] = None, site2: Optional[List] = None, theme2: Optional[List] = None, theme4: Optional[List] = None, theme: Optional[List] = None, site: Optional[List] = None, theme3: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.site2 = site2 if site2 is not None else []
        self.theme2 = theme2 if theme2 is not None else []
        self.theme4 = theme4 if theme4 is not None else []
        self.theme = theme if theme is not None else []
        self.site = site if site is not None else []
        self.theme3 = theme3 if theme3 is not None else []
    def __repr__(self):
        return f"Binding(mention='{self.mention}', site2={self.site2}, theme2={self.theme2}, theme4={self.theme4}, theme={self.theme}, site={self.site}, theme3={self.theme3})"

@dataclass
class Localization(Event):
    def __init__(self, mention: Optional[str] = None, toloc: Optional[List] = None, theme: Optional[List] = None, atloc: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.toloc = toloc if toloc is not None else []
        self.theme = theme if theme is not None else []
        self.atloc = atloc if atloc is not None else []
    def __repr__(self):
        return f"Localization(mention='{self.mention}', toloc={self.toloc}, theme={self.theme}, atloc={self.atloc})"

@dataclass
class Negative_regulation(Event):
    def __init__(self, mention: Optional[str] = None, theme: Optional[List] = None, csite: Optional[List] = None, site: Optional[List] = None, cause: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.theme = theme if theme is not None else []
        self.csite = csite if csite is not None else []
        self.site = site if site is not None else []
        self.cause = cause if cause is not None else []
    def __repr__(self):
        return f"Negative_regulation(mention='{self.mention}', theme={self.theme}, csite={self.csite}, site={self.site}, cause={self.cause})"

@dataclass
class Transcription(Event):
    def __init__(self, mention: Optional[str] = None, theme: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.theme = theme if theme is not None else []
    def __repr__(self):
        return f"Transcription(mention='{self.mention}', theme={self.theme})"

@dataclass
class Protein_catabolism(Event):
    def __init__(self, mention: Optional[str] = None, theme: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.theme = theme if theme is not None else []
    def __repr__(self):
        return f"Protein_catabolism(mention='{self.mention}', theme={self.theme})"

