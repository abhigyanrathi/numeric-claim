"""Exceptions shared across the package."""


class ConfigError(Exception):
    """A problem with the configuration, a claims module, the lock file or the command line."""


class ClaimError(Exception):
    """A claim could not be fingerprinted or computed."""
