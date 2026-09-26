from abc import ABC, abstractmethod
from collections.abc import Iterator


class RetailerAdapter(ABC):
    """Common interface for all retailer data sources."""

    @property
    @abstractmethod
    def retailer(self) -> str:
        """Return the retailer name."""

    @abstractmethod
    def products(self) -> Iterator[dict]:
        """Yield standardized product records."""

    @abstractmethod
    def nutrition(self) -> Iterator[dict]:
        """Yield standardized nutrition records."""
