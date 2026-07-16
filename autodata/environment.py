"""Provider-neutral contract for the environment an agent task may use."""
from dataclasses import asdict, dataclass, field
import json


@dataclass
class EnvironmentSpec:
    """Capabilities exposed to generated tasks; never include credentials here."""
    provider: str = "generic"
    name: str = "standalone"
    description: str = "A self-contained environment described by the source material."
    capabilities: list[str] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict):
            raise ValueError("environment contract must be a JSON object")
        allowed = {"provider", "name", "description", "capabilities", "tools", "constraints", "metadata"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown environment fields: {sorted(unknown)}")
        spec = cls(**value)
        if not all(isinstance(item, str) and item.strip() for item in
                   [spec.provider, spec.name, spec.description]):
            raise ValueError("environment provider, name, and description must be non-empty strings")
        if not all(isinstance(item, str) and item.strip() for item in spec.capabilities + spec.constraints):
            raise ValueError("environment capabilities and constraints must be string arrays")
        if not all(isinstance(item, dict) for item in spec.tools):
            raise ValueError("environment tools must be an array of objects")
        if not isinstance(spec.metadata, dict):
            raise ValueError("environment metadata must be an object")
        return spec

    def to_dict(self):
        return asdict(self)

    def prompt(self):
        return "ENVIRONMENT CONTRACT (the only available environment):\n" + json.dumps(
            self.to_dict(), sort_keys=True
        )
