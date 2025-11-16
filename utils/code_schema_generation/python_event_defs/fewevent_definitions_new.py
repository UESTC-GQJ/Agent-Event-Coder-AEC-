from dataclasses import dataclass
from typing import List, Optional
from utils_typing import Entity, Event, Relation, dataclass


@dataclass
class ManufactureEvent:
    pass

@dataclass
class PersonnelEvent:
    pass

@dataclass
class ContactEvent:
    pass

@dataclass
class WineEvent:
    pass

@dataclass
class MovementEvent:
    pass

@dataclass
class MilitaryEvent:
    pass

@dataclass
class OrganizationEvent:
    pass

@dataclass
class BusinessEvent:
    pass

@dataclass
class PeopleEvent:
    pass

@dataclass
class ConflictEvent:
    pass

@dataclass
class JusticeEvent:
    pass

@dataclass
class MusicEvent:
    pass

@dataclass
class OlympicsEvent:
    pass

@dataclass
class TransactionEvent:
    pass

@dataclass
class SportsEvent:
    pass

@dataclass
class ProjectsEvent:
    pass

@dataclass
class LifeEvent:
    pass

@dataclass
class EducationEvent:
    pass

@dataclass
class FilmEvent:
    pass

@dataclass
class Collaboration(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Collaboration(mention='{self.mention}')"

@dataclass
class DeclareBankruptcy(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"DeclareBankruptcy(mention='{self.mention}')"

@dataclass
class Employment(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Employment(mention='{self.mention}')"

@dataclass
class EmploymentTenure(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"EmploymentTenure(mention='{self.mention}')"

@dataclass
class EndOrg(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"EndOrg(mention='{self.mention}')"

@dataclass
class Financing(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Financing(mention='{self.mention}')"

@dataclass
class Investment(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Investment(mention='{self.mention}')"

@dataclass
class Layoff(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Layoff(mention='{self.mention}')"

@dataclass
class MergeOrg(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"MergeOrg(mention='{self.mention}')"

@dataclass
class Sponsorship(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Sponsorship(mention='{self.mention}')"

@dataclass
class StartOrg(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"StartOrg(mention='{self.mention}')"

@dataclass
class StartSubsidiary(BusinessEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"StartSubsidiary(mention='{self.mention}')"

@dataclass
class Attack(ConflictEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Attack(mention='{self.mention}')"

@dataclass
class Demonstrate(ConflictEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Demonstrate(mention='{self.mention}')"

@dataclass
class Riot(ConflictEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Riot(mention='{self.mention}')"

@dataclass
class SelfImmolation(ConflictEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"SelfImmolation(mention='{self.mention}')"

@dataclass
class Broadcast(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Broadcast(mention='{self.mention}')"

@dataclass
class Contact(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Contact(mention='{self.mention}')"

@dataclass
class Correspondence(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Correspondence(mention='{self.mention}')"

@dataclass
class EMail(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"EMail(mention='{self.mention}')"

@dataclass
class LetterCommunication(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"LetterCommunication(mention='{self.mention}')"

@dataclass
class Meet(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Meet(mention='{self.mention}')"

@dataclass
class OnlineChat(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"OnlineChat(mention='{self.mention}')"

@dataclass
class PhoneWrite(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"PhoneWrite(mention='{self.mention}')"

@dataclass
class VideoChat(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"VideoChat(mention='{self.mention}')"

@dataclass
class VoiceMail(ContactEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"VoiceMail(mention='{self.mention}')"

@dataclass
class Education(EducationEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Education(mention='{self.mention}')"

@dataclass
class DubbingPerformance(FilmEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"DubbingPerformance(mention='{self.mention}')"

@dataclass
class FilmCrewGig(FilmEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"FilmCrewGig(mention='{self.mention}')"

@dataclass
class FilmDistribution(FilmEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"FilmDistribution(mention='{self.mention}')"

@dataclass
class FilmFestival(FilmEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"FilmFestival(mention='{self.mention}')"

@dataclass
class FilmProduction(FilmEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"FilmProduction(mention='{self.mention}')"

@dataclass
class Acquit(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Acquit(mention='{self.mention}')"

@dataclass
class Appeal(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Appeal(mention='{self.mention}')"

@dataclass
class ArrestJail(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"ArrestJail(mention='{self.mention}')"

@dataclass
class ChargeIndict(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"ChargeIndict(mention='{self.mention}')"

@dataclass
class Convict(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Convict(mention='{self.mention}')"

@dataclass
class Execute(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Execute(mention='{self.mention}')"

@dataclass
class Extradite(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Extradite(mention='{self.mention}')"

@dataclass
class Fine(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Fine(mention='{self.mention}')"

@dataclass
class Pardon(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Pardon(mention='{self.mention}')"

@dataclass
class ReleaseParole(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"ReleaseParole(mention='{self.mention}')"

@dataclass
class Sentence(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Sentence(mention='{self.mention}')"

@dataclass
class Sue(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Sue(mention='{self.mention}')"

@dataclass
class TrialHearing(JusticeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"TrialHearing(mention='{self.mention}')"

@dataclass
class Abortion(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Abortion(mention='{self.mention}')"

@dataclass
class BeBorn(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"BeBorn(mention='{self.mention}')"

@dataclass
class Die(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Die(mention='{self.mention}')"

@dataclass
class Divorce(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Divorce(mention='{self.mention}')"

@dataclass
class Homesick(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Homesick(mention='{self.mention}')"

@dataclass
class Injure(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Injure(mention='{self.mention}')"

@dataclass
class Marry(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Marry(mention='{self.mention}')"

@dataclass
class Pregnancy(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Pregnancy(mention='{self.mention}')"

@dataclass
class Sick(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Sick(mention='{self.mention}')"

@dataclass
class Travel(LifeEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Travel(mention='{self.mention}')"

@dataclass
class Artifact(ManufactureEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Artifact(mention='{self.mention}')"

@dataclass
class LeanManufacturing(ManufactureEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"LeanManufacturing(mention='{self.mention}')"

@dataclass
class MilitaryCommand(MilitaryEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"MilitaryCommand(mention='{self.mention}')"

@dataclass
class MilitaryService(MilitaryEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"MilitaryService(mention='{self.mention}')"

@dataclass
class RecruitTraining(MilitaryEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"RecruitTraining(mention='{self.mention}')"

@dataclass
class Recruitment(MilitaryEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Recruitment(mention='{self.mention}')"

@dataclass
class Driving(MovementEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Driving(mention='{self.mention}')"

@dataclass
class Parking(MovementEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Parking(mention='{self.mention}')"

@dataclass
class Transport(MovementEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Transport(mention='{self.mention}')"

@dataclass
class TransportArtifact(MovementEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"TransportArtifact(mention='{self.mention}')"

@dataclass
class Transportperson(MovementEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Transportperson(mention='{self.mention}')"

@dataclass
class Compose(MusicEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Compose(mention='{self.mention}')"

@dataclass
class Dance(MusicEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Dance(mention='{self.mention}')"

@dataclass
class GroupMembership(MusicEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"GroupMembership(mention='{self.mention}')"

@dataclass
class Sing(MusicEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Sing(mention='{self.mention}')"

@dataclass
class ClosingCeremony(OlympicsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"ClosingCeremony(mention='{self.mention}')"

@dataclass
class OlympicAthleteAffiliation(OlympicsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"OlympicAthleteAffiliation(mention='{self.mention}')"

@dataclass
class OlympicMedalHonor(OlympicsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"OlympicMedalHonor(mention='{self.mention}')"

@dataclass
class OpeningCeremony(OlympicsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"OpeningCeremony(mention='{self.mention}')"

@dataclass
class DivisionOfLabour(OrganizationEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"DivisionOfLabour(mention='{self.mention}')"

@dataclass
class Leadership(OrganizationEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Leadership(mention='{self.mention}')"

@dataclass
class OrgCommunication(OrganizationEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"OrgCommunication(mention='{self.mention}')"

@dataclass
class OrganizationBoardMembership(OrganizationEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"OrganizationBoardMembership(mention='{self.mention}')"

@dataclass
class Appointment(PeopleEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Appointment(mention='{self.mention}')"

@dataclass
class PlaceLived(PeopleEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"PlaceLived(mention='{self.mention}')"

@dataclass
class Demotion(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Demotion(mention='{self.mention}')"

@dataclass
class Elect(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Elect(mention='{self.mention}')"

@dataclass
class EndPosition(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"EndPosition(mention='{self.mention}')"

@dataclass
class Nominate(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Nominate(mention='{self.mention}')"

@dataclass
class PerformanceAppraisal(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"PerformanceAppraisal(mention='{self.mention}')"

@dataclass
class Promotion(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Promotion(mention='{self.mention}')"

@dataclass
class Resignation(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Resignation(mention='{self.mention}')"

@dataclass
class StartPosition(PersonnelEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"StartPosition(mention='{self.mention}')"

@dataclass
class ProjectParticipation(ProjectsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"ProjectParticipation(mention='{self.mention}')"

@dataclass
class FairPlay(SportsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"FairPlay(mention='{self.mention}')"

@dataclass
class SportsTeamRoster(SportsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"SportsTeamRoster(mention='{self.mention}')"

@dataclass
class SportsTeamSeasonRecord(SportsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"SportsTeamSeasonRecord(mention='{self.mention}')"

@dataclass
class Tournament(SportsEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Tournament(mention='{self.mention}')"

@dataclass
class Deposit(TransactionEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Deposit(mention='{self.mention}')"

@dataclass
class MoneyLaundering(TransactionEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"MoneyLaundering(mention='{self.mention}')"

@dataclass
class Transaction(TransactionEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"Transaction(mention='{self.mention}')"

@dataclass
class TransferMoney(TransactionEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"TransferMoney(mention='{self.mention}')"

@dataclass
class TransferOwnership(TransactionEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"TransferOwnership(mention='{self.mention}')"

@dataclass
class GrapeVarietyComposition(WineEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"GrapeVarietyComposition(mention='{self.mention}')"

@dataclass
class TrackContribution(MusicEvent):
    def __init__(self, mention: Optional[str] = None):
        self.mention = mention if mention is not None else []
    def __repr__(self):
        return f"TrackContribution(mention='{self.mention}')"

