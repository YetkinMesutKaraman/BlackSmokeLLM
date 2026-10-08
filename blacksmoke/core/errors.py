"""Domain errors shared across layers. The API layer maps them to HTTP responses."""


class BlackSmokeError(Exception):
    """Base class for all expected, domain-level failures."""


class ConfigError(BlackSmokeError):
    """Configuration is missing, malformed, or inconsistent with the registered tasks."""


class UnknownTaskError(BlackSmokeError):
    pass


class UnknownModelError(BlackSmokeError):
    """A requested provider/model is not present in the model catalog."""


class DatasetError(BlackSmokeError):
    pass


class DatasetNotFoundError(DatasetError):
    pass


class InvalidDatasetPathError(DatasetError):
    """The requested path resolves outside the configured data directory."""


class DatasetFormatError(DatasetError):
    pass


class NoReviewsError(DatasetError):
    """No reviews are left after selection and filtering."""
