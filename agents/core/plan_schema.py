"""
Plan Schema Validation for CSIP Central and Team Supervisors.

Provides Pydantic models that validate LLM-produced plan JSON before any
Context Graph writes happen.  A ValidationError raised here is treated
identically to a JsonOutputParser failure and triggers the existing retry
logic in create_plan_node().

Usage (central supervisor)::

    from agents.core.plan_schema import ExecutionPlan, validate_central_plan

    plan = planning_chain.invoke(...)          # raw dict from JsonOutputParser
    validated = validate_central_plan(plan)    # raises ValidationError on bad plan
    # proceed to store_plan() only if no exception

Usage (team supervisor)::

    from agents.core.plan_schema import TeamExecutionPlan, validate_team_plan

    plan = planning_chain.invoke(...)
    validated = validate_team_plan(plan)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid agent names (must stay in sync with VALID_REMOTE_AGENTS in
# central_supervisor.py)
# ---------------------------------------------------------------------------
VALID_CENTRAL_AGENTS = {
    "member_services_team",
    "claims_services_team",
    "pa_services_team",
    "provider_services_team",
    "search_services_team",
}


# ---------------------------------------------------------------------------
# Central plan models
# ---------------------------------------------------------------------------

class PlanGoal(BaseModel):
    """A single goal within a central or team execution plan."""
    id: str
    description: str
    priority: int = 1

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal id must not be empty")
        return v.strip()

    @field_validator("priority")
    @classmethod
    def priority_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("goal priority must be >= 1")
        return v


class PlanStep(BaseModel):
    """A single step within a central execution plan."""
    step_id: str
    goal_id: str
    agent: str
    instruction: str = ""
    query: str
    order: int = 1

    @field_validator("step_id", "goal_id", "query")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("step_id, goal_id and query must not be empty")
        return v.strip()

    @field_validator("agent")
    @classmethod
    def agent_valid(cls, v: str) -> str:
        if v not in VALID_CENTRAL_AGENTS:
            raise ValueError(
                f"agent '{v}' is not a recognized team. "
                f"Must be one of: {sorted(VALID_CENTRAL_AGENTS)}"
            )
        return v

    @field_validator("order")
    @classmethod
    def order_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step order must be >= 1")
        return v


class ExecutionPlan(BaseModel):
    """Central execution plan produced by the planning LLM."""
    goals: List[PlanGoal]
    steps: List[PlanStep]

    @field_validator("goals")
    @classmethod
    def at_least_one_goal(cls, v: List[PlanGoal]) -> List[PlanGoal]:
        if not v:
            raise ValueError("plan must contain at least one goal")
        return v

    @field_validator("steps")
    @classmethod
    def at_least_one_step(cls, v: List[PlanStep]) -> List[PlanStep]:
        if not v:
            raise ValueError("plan must contain at least one step")
        return v

    @model_validator(mode="after")
    def steps_reference_valid_goals(self) -> "ExecutionPlan":
        goal_ids = {g.id for g in self.goals}
        for step in self.steps:
            if step.goal_id not in goal_ids:
                raise ValueError(
                    f"step '{step.step_id}' references goal_id '{step.goal_id}' "
                    f"which does not exist in goals {sorted(goal_ids)}"
                )
        return self


# ---------------------------------------------------------------------------
# Team plan models
# ---------------------------------------------------------------------------

class TeamPlanStep(BaseModel):
    """A single step within a team-level execution plan."""
    step_id: str
    goal_id: str
    action: str = ""
    worker: str

    @field_validator("step_id", "goal_id", "worker")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("step_id, goal_id and worker must not be empty")
        return v.strip()


class TeamExecutionPlan(BaseModel):
    """Team execution plan produced by a team supervisor's planning LLM."""
    goals: List[PlanGoal]
    steps: List[TeamPlanStep]

    @field_validator("goals")
    @classmethod
    def at_least_one_goal(cls, v: List[PlanGoal]) -> List[PlanGoal]:
        if not v:
            raise ValueError("team plan must contain at least one goal")
        return v

    @field_validator("steps")
    @classmethod
    def at_least_one_step(cls, v: List[TeamPlanStep]) -> List[TeamPlanStep]:
        if not v:
            raise ValueError("team plan must contain at least one step")
        return v

    @model_validator(mode="after")
    def steps_reference_valid_goals(self) -> "TeamExecutionPlan":
        goal_ids = {g.id for g in self.goals}
        for step in self.steps:
            if step.goal_id not in goal_ids:
                raise ValueError(
                    f"step '{step.step_id}' references goal_id '{step.goal_id}' "
                    f"which does not exist in goals {sorted(goal_ids)}"
                )
        return self


# ---------------------------------------------------------------------------
# Convenience validators
# ---------------------------------------------------------------------------

def validate_central_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a central plan dict produced by JsonOutputParser.

    Returns the original dict unchanged if valid (so callers can keep using
    the dict directly).  Raises pydantic.ValidationError on failure.
    """
    ExecutionPlan.model_validate(raw)
    return raw


def validate_team_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a team plan dict produced by JsonOutputParser.

    Returns the original dict unchanged if valid.
    Raises pydantic.ValidationError on failure.
    """
    TeamExecutionPlan.model_validate(raw)
    return raw
