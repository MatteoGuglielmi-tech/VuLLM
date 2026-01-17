from .typedefs import Messages


class PromptTemplate:
    """Base class for prompt templates—not a dataclass since templates are static."""

    SYSTEM_PROMPT: str = "Override in subclass"
    USER_PROMPT: str = "Override in subclass"

    def __iter__(self):
        return iter([self.SYSTEM_PROMPT, self.USER_PROMPT])

    @property
    def system(self) -> str:
        return self.SYSTEM_PROMPT

    @property
    def user(self) -> str:
        return self.USER_PROMPT

    def as_messages(self, formatted_user_content: str) -> Messages:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_content},
        ]
