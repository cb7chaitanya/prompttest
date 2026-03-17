"""Domain models for prompttest."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class PromptConfig:
    """A versioned prompt template loaded from YAML."""

    name: str
    version: str
    model: str
    provider: str
    system: str
    template: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptConfig:
        return cls(
            name=data["name"],
            version=data.get("version", "1"),
            model=data.get("model", "gpt-4o-mini"),
            provider=data.get("provider", "openai"),
            system=data.get("system", ""),
            template=data["template"],
            parameters=data.get("parameters", {}),
        )


@dataclass
class TestCase:
    """A single input/expected-output pair from a dataset."""

    input: str
    expected: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    """A collection of test cases loaded from YAML."""

    name: str
    prompt: str  # references a prompt config name
    cases: list[TestCase]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        raw_cases = data.get("cases", data.get("tests", []))
        cases = [
            TestCase(
                input=c["input"],
                expected=c["expected"],
                tags=c.get("tags", []),
            )
            for c in raw_cases
        ]
        return cls(name=data["name"], prompt=data["prompt"], cases=cases)


@dataclass
class CaseResult:
    """Result of evaluating a single test case."""

    case: TestCase
    output: str
    verdict: Verdict
    score: float
    reason: str


@dataclass
class RunResult:
    """Aggregated result of a full test run."""

    prompt_name: str
    prompt_version: str
    dataset_name: str
    results: list[CaseResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.FAIL)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0
