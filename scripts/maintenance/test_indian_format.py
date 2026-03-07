"""
Test script to demonstrate Indian currency formatting
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indian_currency import format_indian_currency, format_indian_currency_short

print("=" * 70)
print("INDIAN CURRENCY FORMATTING TEST")
print("=" * 70)

test_values = [
    95.50,
    1361.52,
    1440684.95,
    2333.35,
    95000,
    100000,
    1000000,
    10000000,
    50000000
]

print("\nFull Format:")
print("-" * 70)
for val in test_values:
    print(f"  {val:>15,.2f}  →  {format_indian_currency(val)}")

print("\nShort Format (Lakhs/Crores):")
print("-" * 70)
for val in test_values:
    print(f"  {val:>15,.2f}  →  {format_indian_currency_short(val)}")

print("\n" + "=" * 70)
print("Verifying specific values from the project:")
print("=" * 70)
print(f"  Total revenue at risk: {format_indian_currency(1440684.95)}")
print(f"  Expected revenue loss: {format_indian_currency(1361.52)}")
print(f"  Monthly charges: {format_indian_currency(95.50)}")
print("=" * 70)
