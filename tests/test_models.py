import pytest
from pydantic import ValidationError

import main


def test_receipt_data_allows_all_null_fields():
    """The prompt returns null for fields it isn't sure about - the schema must accept that."""
    data = main.ReceiptData()
    assert data.merchant_name is None
    assert data.total_amount is None
    assert data.tax_breakdown == []


def test_receipt_data_coerces_numeric_strings_to_float():
    """GPT sometimes returns numbers as strings; Pydantic v2 coerces them in lax mode."""
    data = main.ReceiptData(total_amount="125.50", tax_amount="18")
    assert data.total_amount == 125.50
    assert data.tax_amount == 18.0


def test_receipt_data_accepts_full_valid_payload():
    data = main.ReceiptData(
        merchant_name="TEST MARKET",
        tax_id="1234567890",
        date="01.01.2026",
        receipt_no="42",
        total_amount=100.5,
        tax_amount=18.0,
        tax_breakdown=[{"rate": 20, "amount": 18.0}],
    )
    assert data.merchant_name == "TEST MARKET"
    assert data.tax_breakdown[0].rate == 20
    assert data.tax_breakdown[0].amount == 18.0


def test_tax_breakdown_item_rejects_non_numeric_rate():
    with pytest.raises(ValidationError):
        main.TaxBreakdownItem(rate="yirmi", amount="cok")


def test_extract_receipt_response_requires_all_top_level_fields():
    with pytest.raises(ValidationError):
        main.ExtractReceiptResponse(status="success")  # durations, data, etc. missing
