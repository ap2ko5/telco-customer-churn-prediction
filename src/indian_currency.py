"""
indian_currency.py
==================
Utility functions for formatting currency in Indian numbering system.

Indian format uses:
- Lakhs (1,00,000) and Crores (1,00,00,000)
- Grouping: rightmost 3 digits, then groups of 2
"""


def format_indian_currency(amount: float, decimals: int = 2) -> str:
    """
    Format amount in Indian numbering system with ₹ symbol.
    
    Examples:
        1440684.95 → ₹14,40,684.95
        10000000 → ₹1,00,00,000.00
        1234.56 → ₹1,234.56
    """
    # Handle negative numbers
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    
    # Split into integer and decimal parts
    if decimals > 0:
        formatted = f"{amount:.{decimals}f}"
        integer_part, decimal_part = formatted.split(".")
    else:
        integer_part = str(int(amount))
        decimal_part = ""
    
    # Apply Indian grouping to integer part
    integer_part = _indian_grouping(integer_part)
    
    # Combine parts
    result = f"₹{sign}{integer_part}"
    if decimal_part:
        result += f".{decimal_part}"
    
    return result


def format_indian_currency_short(amount: float) -> str:
    """
    Format with L (lakhs) or Cr (crores) suffix for large amounts.
    
    Examples:
        1440684.95 → ₹14.41L
        10000000 → ₹1.00Cr
        50000 → ₹50,000
    """
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    
    if amount >= 10_000_000:  # 1 crore or more
        return f"₹{sign}{amount/10_000_000:.2f}Cr"
    elif amount >= 100_000:  # 1 lakh or more
        return f"₹{sign}{amount/100_000:.2f}L"
    elif amount >= 1_000:  # 1 thousand or more
        return f"₹{sign}{_indian_grouping(str(int(amount)))}"
    else:
        return f"₹{sign}{amount:,.0f}"


def _indian_grouping(num_str: str) -> str:
    """
    Apply Indian digit grouping (rightmost 3, then groups of 2).
    
    Example: "1440684" → "14,40,684"
    """
    # Remove any existing commas
    num_str = num_str.replace(",", "")
    
    if len(num_str) <= 3:
        return num_str
    
    # Take rightmost 3 digits
    result = num_str[-3:]
    remaining = num_str[:-3]
    
    # Group remaining digits by 2 from right
    while remaining:
        if len(remaining) <= 2:
            result = remaining + "," + result
            break
        else:
            result = remaining[-2:] + "," + result
            remaining = remaining[:-2]
    
    return result
