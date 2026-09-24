import unittest

from calculator import calculate_total


class CalculatorTests(unittest.TestCase):
    def test_percentage_tax_is_applied(self):
        self.assertEqual(calculate_total(100.0, 0.08), 108.0)

    def test_result_is_rounded_for_currency(self):
        self.assertEqual(calculate_total(12.47, 0.0825), 13.50)


if __name__ == "__main__":
    unittest.main()

