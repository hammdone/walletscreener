# Ethereum Wallet Risk Screener

## Overview
This Flask-based web app analyzes Ethereum wallet addresses to assess their risk level based on blockchain activity and known intelligence (e.g., mixers, scams, centralized exchanges). It fetches live data via the Etherscan API, extracts behavioral features, and applies a trained machine learning model to classify wallets as **RISKY** or **SAFE**.

---

## Features
- Real-time Ethereum transaction and token analysis
- Machine learning model (RandomForest) for risk prediction
- Risk scoring system based on multiple risk categories
- Explanation of key risk factors for flagged wallets
- Frontend chat-style UI to enter wallet addresses and receive results

---

## Setup

### Prerequisites
- Python 3.8+
- An Etherscan API key (set as environment variable or configure in app.py)

### Installation

1. Clone the repo:
```bash
git clone https://github.com/yourusername/wallet-risk-screener.git
cd wallet-risk-screener
