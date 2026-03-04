from typing import Type, TypeVar

from .base import PromptTemplate

T = TypeVar("T", bound="PromptTemplate")


class PromptTemplateFactory:
    """Factory for creating prompt templates."""

    _templates: dict[str, Type["PromptTemplate"]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a template class.

        Examples
        --------
        >>> @PromptTemplateFactory.register("cwe")
        ... class CWEPromptTemplate(PromptTemplate):
        ...     pass
        """

        def decorator(template_class: Type[T]) -> Type[T]:
            if name in cls._templates:
                raise ValueError(f"Template '{name}' is already registered")
            cls._templates[name] = template_class
            return template_class

        return decorator

    @classmethod
    def create(cls, name: str) -> "PromptTemplate":
        """
        Create a template instance by name.

        Parameters
        ----------
        name : str
            Name of the registered template

        Returns
        -------
        PromptTemplate
            Instance of the requested template

        Raises
        ------
        ValueError
            If template name is not registered
        """
        template_class = cls._templates.get(name)
        if template_class is None:
            available = ", ".join(sorted(cls._templates.keys()))
            raise ValueError(
                f"Unknown template: '{name}'. " f"Available templates: {available}"
            )
        return template_class()

    @classmethod
    def list_templates(cls) -> list[str]:
        """List all registered template names."""
        return sorted(cls._templates.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a template is registered."""
        return name in cls._templates
