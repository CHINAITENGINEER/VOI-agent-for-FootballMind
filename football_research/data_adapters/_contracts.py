"""Compatibility imports from the now-ready public platform package.

Round two removed the private importlib loader after the platform repaired its
root imports. Keeping this tiny re-export preserves existing research imports
while every consumer now shares the actual public type identities.
"""
from football_spider.models.contracts import (
    Contract, SCHEMA_VERSION, Observation, EntityRef, MatchRecord, Quote, Score,
)

__all__ = ["Contract", "SCHEMA_VERSION", "Observation", "EntityRef", "MatchRecord", "Quote", "Score"]
