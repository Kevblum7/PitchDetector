"""Pitcher (a.k.a. project) endpoints — CLAUDE.md §16.

A "project" in the product sense maps to a single ``Pitcher``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from backend.app.db.models import Pitcher
from backend.app.db.session import get_session
from backend.app.schemas.requests import PitcherCreate

router = APIRouter(prefix="/api/v1/pitchers", tags=["pitchers"])


@router.post("", response_model=Pitcher, status_code=status.HTTP_201_CREATED)
def create_pitcher(body: PitcherCreate, session: Session = Depends(get_session)) -> Pitcher:
    pitcher = Pitcher(name=body.name, throws=body.throws, notes=body.notes)
    session.add(pitcher)
    session.commit()
    session.refresh(pitcher)
    return pitcher


@router.get("", response_model=list[Pitcher])
def list_pitchers(session: Session = Depends(get_session)) -> list[Pitcher]:
    return list(session.exec(select(Pitcher)).all())


@router.get("/{pitcher_id}", response_model=Pitcher)
def get_pitcher(pitcher_id: int, session: Session = Depends(get_session)) -> Pitcher:
    pitcher = session.get(Pitcher, pitcher_id)
    if pitcher is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"pitcher {pitcher_id} not found")
    return pitcher
