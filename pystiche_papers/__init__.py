import warnings

try:
    from ._version import version as __version__  # type: ignore[import]
except ImportError:
    warnings.warn("version file not found")
    __version__ = "UNKNOWN"
