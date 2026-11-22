"""physics-studio: a robot plus a plain-language task becomes a narrated physics simulation clip."""

__version__ = "0.1.0"


class RecipeError(ValueError):
    """A recipe, robot file or option that cannot be used. The message says what to fix."""
