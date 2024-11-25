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

# Functional Tests

class FinTrackFunctionalTests(unittest.TestCase):

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

    def test_add_and_remove_asset_functionality(self):
        """Functional test for adding and removing assets from the portfolio."""
        new_asset = {"name": "TSLA", "volatility": 1.8}
        
        # Add asset and verify it exists in the portfolio
        self.portfolio_manager.add_asset(new_asset)
        self.assertIn(new_asset, self.portfolio_manager.portfolio)

        # Remove asset and verify it is no longer in the portfolio
        self.portfolio_manager.remove_asset(new_asset)
        self.assertNotIn(new_asset, self.portfolio_manager.portfolio)

    def test_risk_calculation_with_portfolio_data(self):
        """Functional test for calculating risk based on portfolio data."""
        self.portfolio_manager.portfolio = self.portfolio
        calculated_risk = self.risk_assessment.calculate_risk(self.portfolio_manager.portfolio)
        self.assertAlmostEqual(calculated_risk, 1.266, places=2)

    def test_data_aggregation_and_market_data_fetch(self):
        """Functional test for fetching market data."""
        price_aapl = self.data_aggregator.fetch_market_data("AAPL")
        price_goog = self.data_aggregator.fetch_market_data("GOOG")
        
        # Assert that fetched data matches expected values
        self.assertEqual(price_aapl, 150.0)
        self.assertEqual(price_goog, 2800.0)
        
        # Test fetching data for a non-existent ticker
        self.assertIsNone(self.data_aggregator.fetch_market_data("UNKNOWN"))

    def test_recommendation_generation(self):
        """Functional test for recommendation generation based on risk tolerance."""
        low_risk_recommendation = self.recommendation_engine.generate_recommendation(self.portfolio, "low")
        high_risk_recommendation = self.recommendation_engine.generate_recommendation(self.portfolio, "high")
        
        self.assertEqual(low_risk_recommendation, "Reduce high-volatility assets")
        self.assertEqual(high_risk_recommendation, "Consider adding growth stocks")

    def test_graph_generation_execution_time(self):
        """Functional test for execution time of graph generation."""
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
