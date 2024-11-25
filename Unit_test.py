import unittest
import time
import matplotlib.pyplot as plt

# Portfolio Management and Analytics Classes

class PortfolioManager:
    def __init__(self):
        self.portfolio = []

    def add_asset(self, asset):
        """Add an asset to the portfolio."""
        self.portfolio.append(asset)
        return self.portfolio

    def remove_asset(self, asset):
        """Remove an asset from the portfolio."""
        if asset in self.portfolio:
            self.portfolio.remove(asset)
        return self.portfolio

class RiskAssessmentModule:
    def calculate_risk(self, portfolio):
        """Calculate portfolio risk as the average volatility of assets."""
        if not portfolio:
            return 0
        return sum(asset['volatility'] for asset in portfolio) / len(portfolio)

class DataAggregator:
    def fetch_market_data(self, ticker):
        """Fetch market data for a given ticker (mocked data)."""
        market_data = {"AAPL": 150.0, "GOOG": 2800.0}
        return market_data.get(ticker, None)

class RecommendationEngine:
    def generate_recommendation(self, portfolio, risk_tolerance):
        """Generate a recommendation based on portfolio risk tolerance."""
        if risk_tolerance == "low":
            return "Reduce high-volatility assets"
        else:
            return "Consider adding growth stocks"

# Placeholder function for generating graphs
def generate_graph(graph_type, data):
    """Simulate the graph generation process (for execution time measurement)."""
    if graph_type == "pie":
        labels = [asset["name"] for asset in data]
        sizes = [asset["volatility"] for asset in data]
        plt.pie(sizes, labels=labels)
    elif graph_type == "line":
        plt.plot([i for i, _ in enumerate(data)], [asset["volatility"] for asset in data])
    elif graph_type == "bar":
        labels = [asset["name"] for asset in data]
        values = [asset["volatility"] for asset in data]
        plt.bar(labels, values)
    elif graph_type == "scatter":
        risks = [asset["volatility"] for asset in data]
        returns = [asset["volatility"] * 1.5 for asset in data]  # Mock returns
        plt.scatter(risks, returns)
    plt.close()  # Close plot to avoid display during tests

# Unit Tests

class FinTrackTests(unittest.TestCase):

    def setUp(self):
        # Initialize classes to be tested
        self.portfolio_manager = PortfolioManager()
        self.risk_assessment = RiskAssessmentModule()
        self.data_aggregator = DataAggregator()
        self.recommendation_engine = RecommendationEngine()
        
        # Example portfolio data
        self.portfolio = [
            {"name": "AAPL", "volatility": 1.2},
            {"name": "GOOG", "volatility": 1.5},
            {"name": "MSFT", "volatility": 1.1}
        ]

    def test_add_asset(self):
        """Test adding an asset to the portfolio."""
        new_asset = {"name": "TSLA", "volatility": 1.8}
        result = self.portfolio_manager.add_asset(new_asset)
        self.assertIn(new_asset, result)

    def test_remove_asset(self):
        """Test removing an asset from the portfolio."""
        asset_to_remove = {"name": "AAPL", "volatility": 1.2}
        self.portfolio_manager.add_asset(asset_to_remove)
        result = self.portfolio_manager.remove_asset(asset_to_remove)
        self.assertNotIn(asset_to_remove, result)

    def test_calculate_risk(self):
        """Test risk calculation function."""
        risk = self.risk_assessment.calculate_risk(self.portfolio)
        self.assertAlmostEqual(risk, 1.266, places=2)

    def test_fetch_market_data(self):
        """Test data fetching from Data Aggregator."""
        price = self.data_aggregator.fetch_market_data("AAPL")
        self.assertEqual(price, 150.0)
        self.assertIsNone(self.data_aggregator.fetch_market_data("UNKNOWN"))

    def test_generate_recommendation(self):
        """Test recommendation generation based on risk tolerance."""
        recommendation_low = self.recommendation_engine.generate_recommendation(self.portfolio, "low")
        recommendation_high = self.recommendation_engine.generate_recommendation(self.portfolio, "high")
        
        self.assertEqual(recommendation_low, "Reduce high-volatility assets")
        self.assertEqual(recommendation_high, "Consider adding growth stocks")

    def test_graph_execution_time(self):
        """Test execution time of graph generation."""
        graph_types = ["pie", "line", "bar", "scatter"]
        execution_times = {}

        for graph_type in graph_types:
            start_time = time.time()
            generate_graph(graph_type, self.portfolio)
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times[graph_type] = execution_time

            print(f"{graph_type.capitalize()} Chart Execution Time: {execution_time:.4f} seconds")

            # Test if execution time is under a threshold (e.g., 1 second for this case)
            self.assertLess(execution_time, 1.0, f"{graph_type.capitalize()} chart took too long!")

# Run Tests
if __name__ == '__main__':
    unittest.main()
