"""
Calculator module with intentional bugs for testing the SWE agent.
"""


class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a, b):
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a, b):
        """Multiply two numbers."""
        # Bug: Using addition instead of multiplication
        result = a + b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a, b):
        """Divide a by b."""
        # Bug: No check for division by zero
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def power(self, base, exponent):
        """Raise base to the power of exponent."""
        # Bug: Logic error - using multiplication instead of exponentiation
        result = base * exponent
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result

    def sqrt(self, number):
        """Calculate square root."""
        # Bug: No check for negative numbers
        result = number**0.5
        self.history.append(f"sqrt({number}) = {result}")
        return result

    def factorial(self, n):
        """Calculate factorial of n."""
        # Bug: Doesn't handle edge cases (n=0, negative numbers)
        if n == 1:
            return 1
        return n * self.factorial(n - 1)

    def get_history(self):
        """Get calculation history."""
        return self.history

    def clear_history(self):
        """Clear calculation history."""
        # Bug: Method doesn't actually clear the history
        pass


def main():
    """Demo function with bugs."""
    calc = Calculator()

    print("Calculator Demo")
    print("-" * 20)

    # Bug: These operations will produce wrong results due to bugs above
    print(f"2 + 3 = {calc.add(2, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"5 * 6 = {calc.multiply(5, 6)}")  # Will print 11 instead of 30
    print(f"8 / 2 = {calc.divide(8, 2)}")
    print(f"2 ^ 3 = {calc.power(2, 3)}")  # Will print 6 instead of 8
    print(f"sqrt(16) = {calc.sqrt(16)}")

    # These will cause errors:
    # print(f"10 / 0 = {calc.divide(10, 0)}")  # Division by zero
    # print(f"sqrt(-4) = {calc.sqrt(-4)}")     # Square root of negative
    # print(f"5! = {calc.factorial(5)}")        # Will cause recursion issues

    print("\nHistory:")
    for item in calc.get_history():
        print(item)


if __name__ == "__main__":
    main()
