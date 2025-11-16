from dataclasses import dataclass
from typing import List, Optional
from utils_typing import Entity, Event, Relation, dataclass


@dataclass
class AttackEvent:
    pass

@dataclass
class VulnerabilityrelatedEvent:
    pass

@dataclass
class Databreach(AttackEvent):
    def __init__(self, mention: Optional[str] = None, compromised_data: Optional[List] = None, tool: Optional[List] = None, attack_pattern: Optional[List] = None, number_of_data: Optional[List] = None, damage_amount: Optional[List] = None, victim: Optional[List] = None, time: Optional[List] = None, number_of_victim: Optional[List] = None, place: Optional[List] = None, attacker: Optional[List] = None, purpose: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.compromised_data = compromised_data if compromised_data is not None else []
        self.tool = tool if tool is not None else []
        self.attack_pattern = attack_pattern if attack_pattern is not None else []
        self.number_of_data = number_of_data if number_of_data is not None else []
        self.damage_amount = damage_amount if damage_amount is not None else []
        self.victim = victim if victim is not None else []
        self.time = time if time is not None else []
        self.number_of_victim = number_of_victim if number_of_victim is not None else []
        self.place = place if place is not None else []
        self.attacker = attacker if attacker is not None else []
        self.purpose = purpose if purpose is not None else []
    def __repr__(self):
        return f"Databreach(mention='{self.mention}', compromised_data={self.compromised_data}, tool={self.tool}, attack_pattern={self.attack_pattern}, number_of_data={self.number_of_data}, damage_amount={self.damage_amount}, victim={self.victim}, time={self.time}, number_of_victim={self.number_of_victim}, place={self.place}, attacker={self.attacker}, purpose={self.purpose})"

@dataclass
class Ransom(AttackEvent):
    def __init__(self, mention: Optional[str] = None, price: Optional[List] = None, payment_method: Optional[List] = None, tool: Optional[List] = None, attack_pattern: Optional[List] = None, damage_amount: Optional[List] = None, victim: Optional[List] = None, time: Optional[List] = None, place: Optional[List] = None, attacker: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.price = price if price is not None else []
        self.payment_method = payment_method if payment_method is not None else []
        self.tool = tool if tool is not None else []
        self.attack_pattern = attack_pattern if attack_pattern is not None else []
        self.damage_amount = damage_amount if damage_amount is not None else []
        self.victim = victim if victim is not None else []
        self.time = time if time is not None else []
        self.place = place if place is not None else []
        self.attacker = attacker if attacker is not None else []
    def __repr__(self):
        return f"Ransom(mention='{self.mention}', price={self.price}, payment_method={self.payment_method}, tool={self.tool}, attack_pattern={self.attack_pattern}, damage_amount={self.damage_amount}, victim={self.victim}, time={self.time}, place={self.place}, attacker={self.attacker})"

@dataclass
class PatchVulnerability(VulnerabilityrelatedEvent):
    def __init__(self, mention: Optional[str] = None, patch: Optional[List] = None, vulnerability: Optional[List] = None, time: Optional[List] = None, vulnerable_system: Optional[List] = None, cve: Optional[List] = None, vulnerable_system_version: Optional[List] = None, issues_addressed: Optional[List] = None, supported_platform: Optional[List] = None, patch_number: Optional[List] = None, releaser: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.patch = patch if patch is not None else []
        self.vulnerability = vulnerability if vulnerability is not None else []
        self.time = time if time is not None else []
        self.vulnerable_system = vulnerable_system if vulnerable_system is not None else []
        self.cve = cve if cve is not None else []
        self.vulnerable_system_version = vulnerable_system_version if vulnerable_system_version is not None else []
        self.issues_addressed = issues_addressed if issues_addressed is not None else []
        self.supported_platform = supported_platform if supported_platform is not None else []
        self.patch_number = patch_number if patch_number is not None else []
        self.releaser = releaser if releaser is not None else []
    def __repr__(self):
        return f"PatchVulnerability(mention='{self.mention}', patch={self.patch}, vulnerability={self.vulnerability}, time={self.time}, vulnerable_system={self.vulnerable_system}, cve={self.cve}, vulnerable_system_version={self.vulnerable_system_version}, issues_addressed={self.issues_addressed}, supported_platform={self.supported_platform}, patch_number={self.patch_number}, releaser={self.releaser})"

@dataclass
class DiscoverVulnerability(VulnerabilityrelatedEvent):
    def __init__(self, mention: Optional[str] = None, capabilities: Optional[List] = None, vulnerability: Optional[List] = None, discoverer: Optional[List] = None, vulnerable_system_owner: Optional[List] = None, time: Optional[List] = None, vulnerable_system: Optional[List] = None, cve: Optional[List] = None, supported_platform: Optional[List] = None, vulnerable_system_version: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.capabilities = capabilities if capabilities is not None else []
        self.vulnerability = vulnerability if vulnerability is not None else []
        self.discoverer = discoverer if discoverer is not None else []
        self.vulnerable_system_owner = vulnerable_system_owner if vulnerable_system_owner is not None else []
        self.time = time if time is not None else []
        self.vulnerable_system = vulnerable_system if vulnerable_system is not None else []
        self.cve = cve if cve is not None else []
        self.supported_platform = supported_platform if supported_platform is not None else []
        self.vulnerable_system_version = vulnerable_system_version if vulnerable_system_version is not None else []
    def __repr__(self):
        return f"DiscoverVulnerability(mention='{self.mention}', capabilities={self.capabilities}, vulnerability={self.vulnerability}, discoverer={self.discoverer}, vulnerable_system_owner={self.vulnerable_system_owner}, time={self.time}, vulnerable_system={self.vulnerable_system}, cve={self.cve}, supported_platform={self.supported_platform}, vulnerable_system_version={self.vulnerable_system_version})"

@dataclass
class Phishing(AttackEvent):
    def __init__(self, mention: Optional[str] = None, tool: Optional[List] = None, attack_pattern: Optional[List] = None, damage_amount: Optional[List] = None, victim: Optional[List] = None, time: Optional[List] = None, place: Optional[List] = None, trusted_entity: Optional[List] = None, attacker: Optional[List] = None, purpose: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.tool = tool if tool is not None else []
        self.attack_pattern = attack_pattern if attack_pattern is not None else []
        self.damage_amount = damage_amount if damage_amount is not None else []
        self.victim = victim if victim is not None else []
        self.time = time if time is not None else []
        self.place = place if place is not None else []
        self.trusted_entity = trusted_entity if trusted_entity is not None else []
        self.attacker = attacker if attacker is not None else []
        self.purpose = purpose if purpose is not None else []
    def __repr__(self):
        return f"Phishing(mention='{self.mention}', tool={self.tool}, attack_pattern={self.attack_pattern}, damage_amount={self.damage_amount}, victim={self.victim}, time={self.time}, place={self.place}, trusted_entity={self.trusted_entity}, attacker={self.attacker}, purpose={self.purpose})"

