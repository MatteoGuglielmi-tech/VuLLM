from abc import ABC, abstractmethod
from pathlib import Path
from jinja2 import Template

from ..datatypes import PromptPhase, AssumptionMode, Messages


class BaseVulnPrompt(ABC):
    phase: PromptPhase
    assumptions_mode: AssumptionMode
    add_cwe_guidelines: bool
    debug_mode: bool = False

    @abstractmethod
    def format_system_prompt(self) -> str: ...

    @abstractmethod
    def format_user_prompt(self, func_code: str) -> str: ...

    @abstractmethod
    def set_guidelines(self) -> str | None: ... 

    @property
    @abstractmethod
    def OPTIMISTIC_ASSUMPTIONS(self) -> str: ...

    @property
    @abstractmethod
    def PESSIMISTIC_ASSUMPTIONS(self) -> str: ...

    @property
    @abstractmethod
    def OPTIMISTIC_REMINDER(self) -> str: ...

    @property
    @abstractmethod
    def PESSIMISTIC_REMINDER(self) -> str: ...

    @property
    @abstractmethod
    def SYSTEM_PROMPT(self) -> Template: ...

    @property
    @abstractmethod
    def USER_PROMPT(self) -> Template: ...

    def __iter__(self):
        """Allow tuple unpacking: system, user = VulnerabilityPromptConfig()"""
        return iter([self.SYSTEM_PROMPT, self.USER_PROMPT])

    def get_assumptions(self) -> str | None:
        """Get assumptions text for the given mode."""
        match self.assumptions_mode:
            case AssumptionMode.OPTIMISTIC:
                return self.OPTIMISTIC_ASSUMPTIONS
            case AssumptionMode.PESSIMISTIC:
                return self.PESSIMISTIC_ASSUMPTIONS
            case AssumptionMode.NONE:
                return None

    def get_reminder(self) -> str | None:
        """Get reminder text for the given mode."""
        match self.assumptions_mode:
            case AssumptionMode.OPTIMISTIC:
                return self.OPTIMISTIC_REMINDER
            case AssumptionMode.PESSIMISTIC:
                return self.PESSIMISTIC_REMINDER
            case AssumptionMode.NONE:
                return None

    def get_assumption_reminder_combo(self) -> tuple[str | None, str | None]:
        if self.phase.needs_assumptions_reminder:
            assumption = self.get_assumptions()
            reminder = self.get_reminder()
            return (assumption, reminder)
        return (None, None)

    def as_messages(self, func_code: str, ground_truth: str | None = None) -> Messages:
        """Create chat messages.

        Parameters
        ----------
        func_code : str
            Function code to analyze
        ground_truth : str | None
            Ground truth JSON response (for training)

        Returns
        -------
        list[dict[str, str]]
            Chat messages in format [{"role": "system", "content": ...}, ...]
        """

        system_prompt = self.format_system_prompt()
        user_prompt = self.format_user_prompt(func_code=func_code)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.phase.is_trainig and ground_truth is not None:
            messages.append({"role": "assistant", "content": ground_truth})

        if self.debug_mode:
            # I know, the file is overwritten at every entry
            self._serialize_in_use_prompt(
                sys_prompt=system_prompt,
                user_prompt=user_prompt,
                ground_truth=ground_truth,
            )

        return messages

    def _serialize_in_use_prompt(
        self,
        sys_prompt: str,
        user_prompt: str,
        ground_truth: str | None,
        dst: str = "prompt",
    ):

        parent_dir = Path(dst)
        parent_dir.mkdir(parents=True, exist_ok=True)
        filename: str = (
            f"{self.phase}_{self.assumptions_mode}{"_guidelines" if self.add_cwe_guidelines else ""}.txt"
        )
        with open(file=parent_dir / filename, mode="w") as text_file:
            text_file.write(50 * "=" + " System prompt " + 50 * "=")
            text_file.write("\n")
            text_file.write(sys_prompt)
            text_file.write("\n\n")
            text_file.write(50 * "=" + " User prompt " + 50 * "=")
            text_file.write("\n")
            text_file.write(user_prompt)
            text_file.write("\n\n")
            if ground_truth is not None:
                text_file.write(50 * "=" + " Assistant prompt " + 50 * "=")
                text_file.write("\n")
                text_file.write(ground_truth)
