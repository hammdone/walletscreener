from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import requests
import datetime
import pandas as pd
import re
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from collections import defaultdict

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("models/wallet_risk_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# Load known risky addresses and mixer flags
df_risky = pd.read_csv('../wallet_screener/dataset/eth_wallets_cleaned.csv')
KNOWN_RISKY_ADDRESSES = set(df_risky.loc[df_risky['label'].str.lower() == 'dodgy', 'address'].str.lower())
MIXER_ADDRESS_MAP = dict(zip(df_risky['address'].str.lower(), df_risky['is_mixer']))

# Known CEX wallets
KNOWN_CEX_ADDRESSES = {
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance
    "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0",  # Kraken
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Bitfinex
    "0x9e927df09a3ff2c3a8e774e1739b3d87896034a5",  # OKX
    "0xdc76cd25977e0a5ae17155770273ad58648900d3",  # Huobi
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13",  # Gemini
    "0x6fc82a5fe25a5cdb58bc74600a40a69c065263f8",  # KuCoin
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Bitstamp
    "0x2b5634c42055806a59e9107ed44d43c426e58258",  # Gate.io
}

# Load sanctioned addresses and bridge contracts
try:
    KNOWN_SANCTIONED_ADDRESSES = set(pd.read_csv("datasets/sanctioned_wallets.csv")["address"].str.lower())
except:
    KNOWN_SANCTIONED_ADDRESSES = set()

KNOWN_BRIDGE_CONTRACTS = {
    "0x5427fefa711eff984124bfbb1ab6fbf5e3da1820",  # Celer Network
    "0x88ad09518695c6c3712ac10a214be5109a655671",  # Multichain
    "0x6b4c7a5e3f0b99fcd83e9c089bfff6eec74ab8c6",  # Across Protocol
    "0x7355efc63ae731f584380a9838292c7046c1e433",  # Synapse
    "0x3a23f943181408eac424116af7b7790c94cb97a5"   # Router Protocol
}

# Features sent to the ML model
MODEL_FEATURE_COLUMNS = [
    "in_cluster_with_risky", "max_daily_tx_count", "small_transfer_count", "idle_days",
    "contract_interactions_count", "time_since_last_tx", "token_diversity", "token_holdings_count",
    "balance_eth", "tx_count", "tx_in_count", "tx_out_count",
    "avg_tx_value", "max_tx_value", "is_mixer"
]

# Additional features only used for risk scoring
RISK_FEATURE_COLUMNS = [
    "sanctions_list_match", "structuring_behavior", "rapid_forwarding",
    "bridge_interaction", "layering_hops"
]

# Explanations for risk features
RISK_FEATURE_EXPLANATIONS = {
    "sanctions_list_match": "Matches known sanctioned address",
    "structuring_behavior": "Shows structuring behavior (smurfing)",
    "rapid_forwarding": "Rapid fund forwarding detected",
    "bridge_interaction": "Interacts with bridge contracts",
    "layering_hops": "Multi-hop transaction pattern"
}

load_dotenv()
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
ETHERSCAN_API_URL = 'https://api.etherscan.io/api'
mixer_detection_cache = {}

# --- Helper Functions ---

def get_transactions(address):
    params = {'module': 'account', 'action': 'txlist', 'address': address,
              'startblock': 0, 'endblock': 99999999, 'sort': 'asc', 'apikey': ETHERSCAN_API_KEY}
    try:
        resp = requests.get(ETHERSCAN_API_URL, params=params, timeout=10)
        data = resp.json()
        return data['result'] if data.get('status') == '1' else []
    except:
        return []

def get_balance(address):
    params = {'module': 'account', 'action': 'balance', 'address': address,
              'tag': 'latest', 'apikey': ETHERSCAN_API_KEY}
    try:
        resp = requests.get(ETHERSCAN_API_URL, params=params, timeout=5)
        data = resp.json()
        return int(data['result']) / 1e18 if data.get('status') == '1' else 0
    except:
        return 0

def get_token_transfers(address):
    params = {'module': 'account', 'action': 'tokentx', 'address': address,
              'startblock': 0, 'endblock': 99999999, 'sort': 'asc', 'apikey': ETHERSCAN_API_KEY}
    try:
        resp = requests.get(ETHERSCAN_API_URL, params=params, timeout=10)
        data = resp.json()
        return data['result'] if data.get('status') == '1' else []
    except:
        return []

def get_token_balance(address, token_contract):
    params = {'module': 'account', 'action': 'tokenbalance',
              'contractaddress': token_contract, 'address': address,
              'tag': 'latest', 'apikey': ETHERSCAN_API_KEY}
    try:
        resp = requests.get(ETHERSCAN_API_URL, params=params, timeout=5)
        data = resp.json()
        return int(data['result']) / 1e18 if data.get('status') == '1' else 0
    except:
        return 0

def get_balances_parallel(address, token_contracts):
    def balance_check(contract):
        return get_token_balance(address, contract)
    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(balance_check, token_contracts))

def check_in_cluster_with_risky(address, txs):
    wallet = address.lower()
    direct_peers = set()
    for tx in txs:
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        if from_addr == wallet:
            direct_peers.add(to_addr)
        elif to_addr == wallet:
            direct_peers.add(from_addr)
    if direct_peers & KNOWN_RISKY_ADDRESSES:
        return 1
    for peer in direct_peers:
        peer_txs = get_transactions(peer)
        for tx in peer_txs:
            if tx.get('from', '').lower() in KNOWN_RISKY_ADDRESSES or tx.get('to', '').lower() in KNOWN_RISKY_ADDRESSES:
                return 1
    return 0

def is_mixer_contract(address):
    addr = address.lower()
    if addr in mixer_detection_cache:
        return mixer_detection_cache[addr]
    params = {'module': 'contract', 'action': 'getsourcecode', 'address': address, 'apikey': ETHERSCAN_API_KEY}
    try:
        resp = requests.get(ETHERSCAN_API_URL, params=params, timeout=5)
        data = resp.json()
        if data.get('status') != '1' or not data.get('result'):
            mixer_detection_cache[addr] = False
            return False
        contract_info = data['result'][0]
        name = contract_info.get('ContractName', '').lower()
        source = contract_info.get('SourceCode', '').lower()
        keywords = ['tornado', 'mixer', 'mixing', 'tornado.cash']
        is_mixer = any(k in name for k in keywords) or any(k in source for k in keywords)
        mixer_detection_cache[addr] = is_mixer
        return is_mixer
    except:
        mixer_detection_cache[addr] = False
        return False

def is_cex_wallet(address):
    return address.lower() in KNOWN_CEX_ADDRESSES

def is_sanctioned_wallet(address):
    return address.lower() in KNOWN_SANCTIONED_ADDRESSES

def detect_structuring(txs, threshold_large=10, small_threshold=0.01):
    for tx in txs:
        value = int(tx["value"]) / 1e18
        if value >= threshold_large:
            outgoing_small = sum(
                1 for t in txs
                if t["from"].lower() == tx["to"].lower() and int(t["value"]) / 1e18 < small_threshold
            )
            if outgoing_small >= 5:
                return 1
    return 0

def detect_rapid_forwarding(address, txs, hours=24):
    received, sent = defaultdict(float), defaultdict(float)
    for tx in txs:
        ts, value = int(tx["timeStamp"]), int(tx["value"]) / 1e18
        if tx["to"].lower() == address.lower(): received[ts] += value
        elif tx["from"].lower() == address.lower(): sent[ts] += value
    for r_time, r_value in received.items():
        sent_after = sum(v for t, v in sent.items() if t > r_time and t - r_time <= hours * 3600)
        if sent_after >= 0.8 * r_value:
            return 1
    return 0

def interacts_with_bridge(txs):
    return any(tx["to"].lower() in KNOWN_BRIDGE_CONTRACTS for tx in txs)

def detect_layering(address, txs):
    peers = {tx["to"].lower() for tx in txs if tx["from"].lower() == address.lower()}
    hop_count = 0
    for peer in peers:
        peer_txs = get_transactions(peer)
        if peer_txs:
            hop_count += 1
    return hop_count

# --- Feature Extraction ---

def extract_features_with_summary(address):
    txs = get_transactions(address)
    if not txs:
        return None, None, None

    wallet_lower = address.lower()
    tx_dates, tx_values = [], []
    daily_tx_counts = {}
    tx_out_count, tx_in_count, contract_interactions = 0, 0, 0
    small_transfer_count = 0
    unique_counterparties = set()
    total_incoming_value = 0
    total_outgoing_value = 0

    for tx in txs:
        ts = int(tx.get('timeStamp', 0))
        day = datetime.datetime.utcfromtimestamp(ts).date()
        daily_tx_counts[day] = daily_tx_counts.get(day, 0) + 1
        tx_dates.append(ts)

        value = int(tx.get('value', 0)) / 1e18
        tx_values.append(value)

        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()

        if from_addr == wallet_lower:
            tx_out_count += 1
            total_outgoing_value += value
            if to_addr != wallet_lower:
                contract_interactions += 1
                unique_counterparties.add(to_addr)
        if to_addr == wallet_lower:
            tx_in_count += 1
            total_incoming_value += value
            unique_counterparties.add(from_addr)

        if value < 0.01:
            small_transfer_count += 1

    now_ts = datetime.datetime.utcnow().timestamp()
    last_tx_ts = max(tx_dates)
    first_tx_ts = min(tx_dates)
    time_since_last_tx = (now_ts - last_tx_ts) / 86400
    idle_days = time_since_last_tx
    max_daily_tx_count = max(daily_tx_counts.values()) if daily_tx_counts else 0
    tx_count = len(txs)
    avg_tx_value = np.mean(tx_values) if tx_values else 0
    max_tx_value = max(tx_values) if tx_values else 0
    activity_days = (last_tx_ts - first_tx_ts) / 86400 if last_tx_ts > first_tx_ts else 0

    token_txs = get_token_transfers(address)
    token_contracts = set(ttx['contractAddress'].lower() for ttx in token_txs)
    balances = get_balances_parallel(address, token_contracts)
    token_diversity = len(token_contracts)
    token_holdings_count = sum(1 for b in balances if b > 0)

    in_cluster_with_risky = check_in_cluster_with_risky(wallet_lower, txs)
    balance_eth = get_balance(address)

    is_mixer_known = MIXER_ADDRESS_MAP.get(wallet_lower, 0)
    is_mixer_dynamic = is_mixer_contract(wallet_lower) if not is_mixer_known else bool(is_mixer_known)
    is_mixer = int(is_mixer_dynamic)

    # Features for model prediction (original features)
    model_features = [
        in_cluster_with_risky, max_daily_tx_count, small_transfer_count, idle_days,
        contract_interactions, time_since_last_tx, token_diversity, token_holdings_count,
        balance_eth, tx_count, tx_in_count, tx_out_count,
        avg_tx_value, max_tx_value, is_mixer
    ]

    # Additional features only for risk scoring
    sanctions_list_match = 1 if is_sanctioned_wallet(address) else 0
    structuring_behavior = detect_structuring(txs)
    rapid_forwarding = detect_rapid_forwarding(address, txs)
    bridge_interaction = 1 if interacts_with_bridge(txs) else 0
    layering_hops = detect_layering(address, txs)
    
    risk_features = [
        sanctions_list_match, structuring_behavior, rapid_forwarding,
        bridge_interaction, layering_hops
    ]

    summary = {
        "total_incoming_eth": total_incoming_value,
        "total_outgoing_eth": total_outgoing_value,
        "unique_counterparties": len(unique_counterparties),
        "first_active_date": datetime.datetime.utcfromtimestamp(first_tx_ts).strftime('%Y-%m-%d'),
        "last_active_date": datetime.datetime.utcfromtimestamp(last_tx_ts).strftime('%Y-%m-%d'),
        "tx_count": tx_count,
        "activity_days": activity_days,
        "sanctions_match": bool(sanctions_list_match),
        "bridge_interaction": bool(bridge_interaction),
        "layering_hops": layering_hops,
        "structuring_behavior": bool(structuring_behavior),
        "rapid_forwarding": bool(rapid_forwarding)
    }

    return model_features, risk_features, summary

def calculate_risk_score(model_features, risk_features, summary, wallet):
    risk = 0
    triggered_features = []
    
    if wallet.lower() in KNOWN_CEX_ADDRESSES:
        return 0, []

    tx_count = model_features[9]
    activity_days = summary.get("activity_days", 0)
    
    if tx_count > 5000 and activity_days < 120:
        return 0, []

    if tx_count <= 3 and activity_days <= 1:
        risk += 40
        triggered_features.append(("new_wallet", "New wallet with few transactions"))

    # Original risk factors from model features
    if model_features[0] == 1: 
        risk += 20
        triggered_features.append(("cluster_risky", "Connected to flagged wallets"))
    
    risk += min(model_features[2] / 50, 1) * 20
    if model_features[2] > 10:
        triggered_features.append(("small_transfers", "Frequent small transfers"))
    
    if model_features[14] == 1: 
        risk += 40
        triggered_features.append(("mixer", "Linked to known mixer"))
    
    # New risk factors from risk_features
    if risk_features[0] == 1: 
        risk += 35
        triggered_features.append(("sanctions", RISK_FEATURE_EXPLANATIONS["sanctions_list_match"]))
    
    if risk_features[1] == 1: 
        risk += 25
        triggered_features.append(("structuring", RISK_FEATURE_EXPLANATIONS["structuring_behavior"]))
    
    if risk_features[2] == 1: 
        risk += 20
        triggered_features.append(("rapid_forwarding", RISK_FEATURE_EXPLANATIONS["rapid_forwarding"]))
    
    if risk_features[3] == 1: 
        risk += 15
        triggered_features.append(("bridge", RISK_FEATURE_EXPLANATIONS["bridge_interaction"]))
    
    layering_risk = min(risk_features[4] * 2, 20)
    risk += layering_risk
    if risk_features[4] > 2:
        triggered_features.append(("layering", f"{RISK_FEATURE_EXPLANATIONS['layering_hops']} ({risk_features[4]} hops)"))

    return round(min(risk, 100), 2), triggered_features

# --- Flask Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_msg = request.json.get('message', '').strip().lower()

        if any(word in user_msg for word in ['hi', 'hello', 'hey', 'salaam', 'gm', 'yo']):
            return jsonify({'reply': "Hello! Paste an Ethereum wallet address to check its risk level."})
        if 'help' in user_msg:
            return jsonify({'reply': "I analyze Ethereum wallet addresses for risk using blockchain activity."})

        match = re.search(r"0x[a-fA-F0-9]{40}", user_msg)
        if not match:
            return jsonify({'reply': "Please provide a valid Ethereum address."})

        wallet = match.group(0).lower()

        # Skip processing if wallet is a known centralized exchange
        if wallet in KNOWN_CEX_ADDRESSES:
            reply_text = (
                "<b>Prediction:</b> <span style='color:green; font-weight:bold'>SAFE</span><br>"
                "<b>Confidence:</b> 1.00<br>"
                "<br><b>Factors:</b><br>"
                "- Known centralized exchange wallet<br>"
            )
            return jsonify({
                'reply': reply_text,
                'risk_scores': None,
            })

        # Extract features and summary
        model_features, risk_features, summary = extract_features_with_summary(wallet)
        if model_features is None:
            return jsonify({'reply': f"Could not extract features for `{wallet}`. Is it inactive or empty?"})

        # Make prediction using only model features
        features_df = pd.DataFrame([model_features], columns=MODEL_FEATURE_COLUMNS)
        probas = model.predict_proba(features_df)[0]
        label_idx = np.argmax(probas)
        raw_label = label_encoder.inverse_transform([label_idx])[0].lower()

        label = "RISKY" if raw_label == "dodgy" else "SAFE"
        score = probas[label_idx]
        risk_score, triggered_features = calculate_risk_score(model_features, risk_features, summary, wallet)

        # Prepare explanations
        explanation = []
        triggered_risk_features = []

        if label == "RISKY":
            # Model-based explanations
            if model_features[0] == 1:
                explanation.append("- Connected to flagged wallets")
            if model_features[2] > 10:
                explanation.append("- Frequent small transfers")
            if model_features[3] < 30:
                explanation.append("- Recently active or not idle")
            if model_features[4] < 10:
                explanation.append("- Low contract usage")
            if model_features[5] < 7:
                explanation.append("- Very recent transaction")
            if model_features[6] < 3:
                explanation.append("- Low token diversity")
            if model_features[7] > 50:
                explanation.append("- High token holdings (possible airdrops or spam)")
            if model_features[14] == 1:
                explanation.append("- Linked to a known mixer")
            if model_features[8] < 0.01:
                explanation.append("- Very low ETH balance")
            if model_features[9] < 5:
                explanation.append("- Very few transactions")
        else:
            if model_features[0] == 0:
                explanation.append("- Not connected to flagged wallets")
            if model_features[3] > 60:
                explanation.append("- Long activity period")
            if model_features[4] > 50:
                explanation.append("- High contract interaction")
            if model_features[6] > 5:
                explanation.append("- Good token diversity")
            if model_features[8] > 0.5:
                explanation.append("- Reasonable ETH balance")
            if model_features[9] > 20:
                explanation.append("- Active transaction history")

        # Separate risk feature explanations
        for feature_code, description in triggered_features:
            if feature_code in ['sanctions', 'structuring', 'rapid_forwarding', 'bridge', 'layering']:
                triggered_risk_features.append(f"- {description}")

        # Format the response
        risk_text = "<br><b>Factors:</b><br>" + "<br>".join(explanation) if explanation else "No major risk indicators detected."
        
        # Add section for triggered risk features if any
        if triggered_risk_features:
            risk_text += "<br><b>Triggered Risk Indicators:</b><br>" + "<br>".join(triggered_risk_features)

        label_color = "<span style='color:red; font-weight:bold'>RISKY</span>" if label == "RISKY" else "<span style='color:green; font-weight:bold'>SAFE</span>"

        reply_text = (
            f"<b>Prediction:</b> {label_color}<br>"
            f"<b>Confidence:</b> {score:.2f}<br>"
            f"<b>Risk Score:</b> {risk_score}/100<br>"
            f"{risk_text}"
        )

        return jsonify({
            'reply': reply_text,
            'risk_scores': {
                'total_risk_score': risk_score,
                'triggered_features': [f[0] for f in triggered_features],
                'feature_details': triggered_features
            },
            'summary': summary
        })

    except Exception as e:
        return jsonify({'reply': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
