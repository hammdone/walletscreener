# Ethereum Wallet Risk Screener

## Overview
A Flask-based web app that analyzes Ethereum wallet addresses and assesses their risk level based on their activity. It fetches live data using the Etherscan API, extracts behavioral features, and applies a trained machine learning model to classify wallets as **RISKY** or **SAFE**.

---

## Features
- Real-time Ethereum transaction and token analysis
- Machine learning model (RandomForest) for risk prediction
- Risk scoring system based on multiple risk categories
- Explanation of key risk factors for flagged wallets
- Frontend chat-style UI to enter wallet addresses and receive results

