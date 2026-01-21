"""Exception classes for gallium."""


class GalliumError(Exception):
    """Base exception class for all gallium errors.

    All gallium-specific exceptions inherit from this class,
    making it easy to catch any gallium error:

    Example:
        >>> try:
        ...     gallium.grid(images)
        ... except GalliumError as e:
        ...     print(f"Gallium error: {e}")
    """

    pass


class MissingDependencyError(GalliumError, ImportError):
    """Raised when an optional dependency is missing.

    This is raised when attempting to use grid or compare features
    without Pillow installed.

    Example:
        >>> try:
        ...     gallium.grid(images)
        ... except MissingDependencyError as e:
        ...     print(e)
        Pillow is required for grid operations. Install it with: pip install pillow
    """

    pass


class InvalidFilterError(GalliumError, ValueError):
    """Raised when an unsupported filter is passed to find().

    Example:
        >>> try:
        ...     gallium.find(invalid_filter="value")
        ... except InvalidFilterError as e:
        ...     print(e)
        Unsupported filter: 'invalid_filter'. Supported filters: ...
    """

    pass


class SerializationError(GalliumError, TypeError):
    """Raised when params contains non-JSON-serializable data.

    Example:
        >>> try:
        ...     gallium.log(prompt="test", params={"func": lambda x: x})
        ... except SerializationError as e:
        ...     print(e)
        params must be JSON-serializable: ...
    """

    pass
