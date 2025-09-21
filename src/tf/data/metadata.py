"""Utilities for describing futures contracts and universes.

This module provides light-weight data classes that capture the minimal
metadata required by the starter backtester.  The objects are intentionally
simple – they focus on symbol identity, sector grouping, point value and the
information necessary to locate a tradeable time-series from public data
sources such as Yahoo! Finance.

Future project phases can extend these structures with richer attributes
like tick-size, expiration calendars or exchange specific roll logic, but the
goal here is to give the data layer a well-defined schema that downstream code
can reason about today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class ContractMetadata:
    """Describe a single futures contract or proxy time-series.

    Parameters
    ----------
    symbol:
        Internal symbol used throughout the backtester.
    sector:
        High-level sector bucket used for risk aggregation.
    point_value:
        Monetary value of a one point price move for the contract.
    currency:
        Reporting currency for the point value.  Defaults to USD.
    contract_step:
        Minimum contract increment when rounding target positions.  Defaults to 1.
    data_source:
        Identifier for the preferred market data vendor.  ``"yahoo"`` is
        supported out of the box.
    data_symbol:
        Optional vendor specific symbol override.  When omitted and the data
        source is Yahoo the loader falls back to the common ``"<sym>=F"``
        futures convention.
    description:
        Human readable text used for logs or reports.
    """

    symbol: str
    sector: str
    point_value: float
    currency: str = "USD"
    contract_step: float = 1.0
    data_source: str = "yahoo"
    data_symbol: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if not self.symbol:
            raise ValueError("Contract symbol must be provided")
        if self.point_value <= 0:
            raise ValueError("Point value must be positive")
        if not self.sector:
            raise ValueError("Sector must be provided")
        if self.contract_step <= 0:
            raise ValueError("Contract step must be positive")

    @property
    def vendor_symbol(self) -> str:
        """Return the symbol understood by the configured data vendor."""

        if self.data_symbol:
            return self.data_symbol
        if self.data_source.lower() == "yahoo":
            # Yahoo futures commonly use the ``ES=F`` style suffix.
            if self.symbol.endswith("=F"):
                return self.symbol
            return f"{self.symbol}=F"
        return self.symbol

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ContractMetadata":
        """Construct metadata from dictionaries (e.g. YAML or JSON records)."""

        data = dict(payload)
        try:
            symbol = str(data.pop("symbol"))
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError("Contract dictionary missing 'symbol'") from exc
        try:
            point_value = float(data.pop("point_value"))
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError("Contract dictionary missing 'point_value'") from exc
        sector = str(data.pop("sector", ""))
        data_source = str(data.pop("data_source", "yahoo"))
        contract_step = float(data.pop("contract_step", 1.0))
        data_symbol = data.pop("data_symbol", None)
        description = data.pop("description", None)
        currency = str(data.pop("currency", "USD"))
        if data:
            # Surface configuration typos early.
            unknown = ", ".join(sorted(data))
            raise KeyError(f"Unknown contract metadata fields: {unknown}")
        return cls(
            symbol=symbol,
            sector=sector,
            point_value=point_value,
            currency=currency,
            contract_step=contract_step,
            data_source=data_source,
            data_symbol=str(data_symbol) if data_symbol is not None else None,
            description=str(description) if description is not None else None,
        )


@dataclass(slots=True)
class UniverseDefinition:
    """Collection of :class:`ContractMetadata` objects with validation helpers."""

    contracts: Sequence[ContractMetadata] = field(default_factory=list)

    def __post_init__(self) -> None:
        symbols = [c.symbol for c in self.contracts]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Universe contains duplicate contract symbols")

    @classmethod
    def from_payload(
        cls, entries: Iterable[Mapping[str, object] | ContractMetadata]
    ) -> "UniverseDefinition":
        """Build a validated universe from dictionaries or metadata objects."""

        contracts: list[ContractMetadata] = []
        for entry in entries:
            if isinstance(entry, ContractMetadata):
                contracts.append(entry)
            else:
                contracts.append(ContractMetadata.from_dict(entry))
        return cls(contracts)

    def as_dataframe(self) -> "pd.DataFrame":
        """Return the universe attributes as a tidy ``pandas`` dataframe."""

        import pandas as pd  # Local import keeps pandas optional for callers

        data = [
            {
                "symbol": c.symbol,
                "sector": c.sector,
                "point_value": c.point_value,
                "currency": c.currency,
                "contract_step": c.contract_step,
                "data_source": c.data_source,
                "data_symbol": c.vendor_symbol,
                "description": c.description or "",
            }
            for c in self.contracts
        ]
        return pd.DataFrame(data).set_index("symbol")

    def by_symbol(self) -> MutableMapping[str, ContractMetadata]:
        """Return a dictionary keyed by internal symbol."""

        return {c.symbol: c for c in self.contracts}
