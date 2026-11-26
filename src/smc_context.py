"""N-H04: context-local analyzer overrides for isolated backtests.

The live engine reads module-level singletons (`market_structure`,
`liquidity`, etc.). During a backtest we must not mutate those
singletons' caches — otherwise live reconfirmation observes backtest
state. This module provides:

- A thread/task-local dict of analyzer instances.
- `AnalyzerProxy`, whose attribute access resolves to the current
  context's instance if set, else to the module-level singleton.

`smc_strategy` imports proxies instead of the singletons, so its
methods require no refactoring. Live callers see no change.
"""
from contextvars import ContextVar
from typing import Any, Dict, Optional


_analyzers_ctx: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "smc_analyzer_instances", default=None
)


def _defaults() -> Dict[str, Any]:
    """Lazy import to avoid a circular dependency at module load."""
    from src.market_structure import market_structure
    from src.liquidity import liquidity
    from src.order_blocks import order_blocks
    from src.fair_value_gaps import fair_value_gaps
    from src.premium_discount import premium_discount
    return {
        "market_structure": market_structure,
        "liquidity": liquidity,
        "order_blocks": order_blocks,
        "fair_value_gaps": fair_value_gaps,
        "premium_discount": premium_discount,
    }


class AnalyzerProxy:
    """Resolves to the current-context instance, else the singleton."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __setattr__(self, key: str, value: Any) -> None:
        # Prevent accidental attribute assignment that would bypass
        # the context. Analyzer proxies are read-only.
        raise AttributeError(
            "AnalyzerProxy is read-only; set attributes on the "
            "underlying instance instead."
        )

    def __getattr__(self, name: str) -> Any:
        ctx = _analyzers_ctx.get()
        target = ctx.get(self._name) if ctx else None
        if target is None:
            target = _defaults()[self._name]
        return getattr(target, name)

    def __repr__(self) -> str:
        return f"<AnalyzerProxy {self._name!r}>"


def set_smc_analyzer_context(analyzers: Dict[str, Any]) -> Any:
    """Install backtest-isolated analyzer instances for this thread."""
    return _analyzers_ctx.set(analyzers)


def reset_smc_analyzer_context(token: Any) -> None:
    """Restore the previous analyzer context."""
    _analyzers_ctx.reset(token)