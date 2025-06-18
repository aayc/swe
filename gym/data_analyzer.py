"""
Data analysis tool with logic bugs for testing the SWE agent.
"""

from typing import Any


class DataAnalyzer:
    def __init__(self):
        self.data = []
        self.analysis_cache = {}

    def load_data(self, data: list[Any]) -> None:
        """Load data for analysis."""
        # Bug: Doesn't validate data format or handle None values
        self.data = data
        # Bug: Doesn't clear cache when new data is loaded

    def calculate_mean(self, numbers: list[float]) -> float:
        """Calculate the mean of a list of numbers."""
        # Bug: No check for empty list
        total = sum(numbers)
        return total / len(numbers)

    def calculate_median(self, numbers: list[float]) -> float:
        """Calculate the median of a list of numbers."""
        # Bug: Doesn't handle even-length lists correctly
        sorted_numbers = sorted(numbers)
        middle_index = len(sorted_numbers) // 2
        return sorted_numbers[middle_index]

    def calculate_mode(self, numbers: list[float]) -> float:
        """Calculate the mode of a list of numbers."""
        # Bug: Only returns one mode, doesn't handle multimodal data
        frequency = {}
        for num in numbers:
            frequency[num] = frequency.get(num, 0) + 1

        # Bug: Returns the key instead of handling ties
        return max(frequency, key=frequency.get)

    def calculate_standard_deviation(self, numbers: list[float]) -> float:
        """Calculate standard deviation."""
        mean = self.calculate_mean(numbers)
        # Bug: Using population formula instead of sample formula
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        return variance**0.5

    def find_outliers(
        self, numbers: list[float], threshold: float = 2.0
    ) -> list[float]:
        """Find outliers using standard deviation method."""
        if not numbers:
            return []

        mean = self.calculate_mean(numbers)
        std_dev = self.calculate_standard_deviation(numbers)

        outliers = []
        for num in numbers:
            # Bug: Logic error - should be absolute value
            if (num - mean) / std_dev > threshold:
                outliers.append(num)

        return outliers

    def analyze_categorical_data(self, categories: list[str]) -> dict[str, Any]:
        """Analyze categorical data."""
        # Bug: No handling of None or empty string values
        frequency = {}
        for category in categories:
            frequency[category] = frequency.get(category, 0) + 1

        total = len(categories)
        percentages = {}
        for category, count in frequency.items():
            # Bug: Integer division instead of float division
            percentages[category] = (count / total) * 100

        return {
            "frequency": frequency,
            "percentages": percentages,
            "total": total,
            # Bug: Accessing non-existent method
            "most_common": max(frequency.items(), key=lambda x: x[1])[0],
        }

    def correlation_coefficient(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        # Bug: No check if lists are same length
        n = len(x)

        # Bug: Division by zero if all values are the same
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(xi * xi for xi in x)
        sum_yy = sum(yi * yi for yi in y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = (
            (n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)
        ) ** 0.5

        return numerator / denominator

    def linear_regression(self, x: list[float], y: list[float]) -> dict[str, float]:
        """Perform simple linear regression."""
        # Bug: No validation of input data
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        # Bug: Potential division by zero
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        return {"slope": slope, "intercept": intercept}

    def group_by(
        self, data: list[dict[str, Any]], key: str
    ) -> dict[str, list[dict[str, Any]]]:
        """Group data by a specific key."""
        groups = {}

        for item in data:
            # Bug: No handling if key doesn't exist in item
            group_key = item[key]

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append(item)

        return groups

    def filter_data(
        self, data: list[dict[str, Any]], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Filter data based on criteria."""
        filtered = []

        for item in data:
            match = True
            for key, value in filters.items():
                # Bug: Only handles exact matches, no range or pattern matching
                if item.get(key) != value:
                    match = False
                    break

            if match:
                filtered.append(item)

        return filtered

    def summarize_numeric_column(
        self, data: list[dict[str, Any]], column: str
    ) -> dict[str, float]:
        """Summarize a numeric column."""
        # Bug: No type checking or validation
        values = [item[column] for item in data if column in item]

        if not values:
            return {}

        return {
            "count": len(values),
            "mean": self.calculate_mean(values),
            "median": self.calculate_median(values),
            "mode": self.calculate_mode(values),
            "std_dev": self.calculate_standard_deviation(values),
            "min": min(values),
            "max": max(values),
            # Bug: Using wrong percentile calculation
            "q1": values[len(values) // 4],
            "q3": values[3 * len(values) // 4],
        }


def main():
    """Demo function with data analysis bugs."""
    analyzer = DataAnalyzer()

    # Sample data with intentional issues
    numeric_data = [1, 2, 3, 4, 5, None, "invalid", 100]  # Mixed types
    categorical_data = [
        "A",
        "B",
        "A",
        "C",
        "B",
        None,
        "",
    ]  # Contains None and empty string

    try:
        # These will demonstrate the bugs
        print("Mean calculation:")
        # Bug: Will fail with mixed data types
        mean = analyzer.calculate_mean([1, 2, 3, 4, 5])
        print(f"Mean: {mean}")

        print("\nOutlier detection:")
        outliers = analyzer.find_outliers([1, 2, 3, 4, 5, 100])
        print(f"Outliers: {outliers}")

        print("\nCategorical analysis:")
        cat_analysis = analyzer.analyze_categorical_data(["A", "B", "A", "C"])
        print(f"Analysis: {cat_analysis}")

        print("\nCorrelation (with different length lists):")
        # Bug: Will fail because lists are different lengths
        corr = analyzer.correlation_coefficient([1, 2, 3], [4, 5, 6, 7])
        print(f"Correlation: {corr}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
