"""
Tests for calculator module - some tests will fail due to bugs in the calculator.
"""

import sys
from pathlib import Path

import pytest

# Add the gym directory to the path to import the calculator module
gym_dir = Path(__file__).parent.parent
sys.path.insert(0, str(gym_dir))

from calculator import Calculator


class TestCalculator:
    def setup_method(self):
        """Set up test fixtures."""
        self.calc = Calculator()

    def test_addition(self):
        """Test addition operation."""
        result = self.calc.add(2, 3)
        assert result == 5
        assert "2 + 3 = 5" in self.calc.get_history()

    def test_subtraction(self):
        """Test subtraction operation."""
        result = self.calc.subtract(10, 4)
        assert result == 6
        assert "10 - 4 = 6" in self.calc.get_history()

    def test_multiplication(self):
        """Test multiplication operation - THIS WILL FAIL due to bug."""
        result = self.calc.multiply(5, 6)
        assert result == 30  # Will fail because multiply() uses addition
        assert "5 * 6 = 30" in self.calc.get_history()

    def test_division(self):
        """Test division operation."""
        result = self.calc.divide(8, 2)
        assert result == 4
        assert "8 / 2 = 4.0" in self.calc.get_history()

    def test_division_by_zero(self):
        """Test division by zero - THIS WILL FAIL due to missing error handling."""
        with pytest.raises(ZeroDivisionError):
            self.calc.divide(10, 0)

    def test_power(self):
        """Test power operation - THIS WILL FAIL due to bug."""
        result = self.calc.power(2, 3)
        assert result == 8  # Will fail because power() uses multiplication
        assert "2 ^ 3 = 8" in self.calc.get_history()

    def test_square_root(self):
        """Test square root operation."""
        result = self.calc.sqrt(16)
        assert result == 4.0
        assert "sqrt(16) = 4.0" in self.calc.get_history()

    def test_square_root_negative(self):
        """Test square root of negative number - THIS WILL FAIL due to missing validation."""
        with pytest.raises(ValueError):
            self.calc.sqrt(-4)

    def test_factorial(self):
        """Test factorial operation - THIS WILL FAIL due to bugs."""
        result = self.calc.factorial(5)
        assert result == 120

    def test_factorial_zero(self):
        """Test factorial of zero - THIS WILL FAIL due to missing edge case handling."""
        result = self.calc.factorial(0)
        assert result == 1

    def test_factorial_negative(self):
        """Test factorial of negative number - THIS WILL FAIL due to missing validation."""
        with pytest.raises(ValueError):
            self.calc.factorial(-1)

    def test_history_tracking(self):
        """Test that operations are properly tracked in history."""
        self.calc.add(1, 2)
        self.calc.subtract(5, 3)

        history = self.calc.get_history()
        assert len(history) == 2
        assert "1 + 2 = 3" in history
        assert "5 - 3 = 2" in history

    def test_clear_history(self):
        """Test clearing history - THIS WILL FAIL due to bug in clear_history()."""
        self.calc.add(1, 2)
        self.calc.clear_history()

        history = self.calc.get_history()
        assert len(history) == 0  # Will fail because clear_history() doesn't work

    def test_chained_operations(self):
        """Test multiple operations in sequence."""
        result1 = self.calc.add(2, 3)
        result2 = self.calc.multiply(result1, 4)  # Will use wrong operation due to bug
        result3 = self.calc.divide(result2, 2)

        # This test will fail due to the multiply bug
        assert result1 == 5
        assert result2 == 20  # Will fail
        assert result3 == 10  # Will fail

    def test_edge_cases(self):
        """Test edge cases with zero and one."""
        assert self.calc.add(0, 0) == 0
        assert self.calc.multiply(5, 0) == 0  # Will fail due to multiply bug
        assert self.calc.multiply(1, 7) == 7  # Will fail due to multiply bug


if __name__ == "__main__":
    pytest.main([__file__])
