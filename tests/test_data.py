import unittest
import sys
import os

# Add the correct path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingestion import fetch_data


class TestData(unittest.TestCase):

    def test_fetch_data(self):
        df = fetch_data("AAPL", limit=100)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn('close', df.columns)


if __name__ == '__main__':
    unittest.main()