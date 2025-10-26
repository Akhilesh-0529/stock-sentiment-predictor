from setuptools import setup, find_packages

setup(
    name="stock-sentiment-predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.0",
        "joblib>=1.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "requests>=2.31.0",
    ],
)