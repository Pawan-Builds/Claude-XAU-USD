"""Phase 1 / C-02 regression: uncertain retcodes are classified."""

import MetaTrader5 as mt5

from src.mt5_connector import MT5Connector, OrderResult


def test_uncertain_retcodes_include_error_timeout_connection():
    codes = MT5Connector._UNCERTAIN_RETCODES
    assert mt5.TRADE_RETCODE_ERROR in codes
    assert mt5.TRADE_RETCODE_TIMEOUT in codes
    assert mt5.TRADE_RETCODE_CONNECTION in codes


def test_definitely_rejected_codes_not_uncertain():
    codes = MT5Connector._UNCERTAIN_RETCODES
    for rc in (
        mt5.TRADE_RETCODE_INVALID,
        mt5.TRADE_RETCODE_INVALID_VOLUME,
        mt5.TRADE_RETCODE_INVALID_PRICE,
        mt5.TRADE_RETCODE_INVALID_STOPS,
        mt5.TRADE_RETCODE_MARKET_CLOSED,
        mt5.TRADE_RETCODE_NO_MONEY,
        mt5.TRADE_RETCODE_PRICE_CHANGED,
        mt5.TRADE_RETCODE_REQUOTE,
    ):
        assert rc not in codes


def test_order_result_uncertain_defaults_false():
    assert OrderResult(success=True).uncertain is False
    assert OrderResult(success=False).uncertain is False


def test_order_result_uncertain_can_be_set():
    assert OrderResult(success=False, uncertain=True).uncertain is True