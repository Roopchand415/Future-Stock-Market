"""
sentiment.py - Financial News Sentiment Analysis Engine
Implements a financial VADER-style rule-based NLP pipeline:
- Context-aware financial lexicon (handles idioms like 'record high', 'low unemployment', 'rate cut')
- Multi-token negation windowing & intensity boosters ('massively', 'sharply', 'marginally')
- In-memory 15-minute TTL caching to avoid redundant Google News network requests
"""

import urllib.parse
import xml.etree.ElementTree as ET
import requests
import re
from services.cache import sentiment_cache

# Financial idioms and multi-word phrases with fixed sentiment weights
FINANCIAL_PHRASES = {
    'all-time high': 2.5,
    'record high': 2.5,
    'beat expectations': 2.0,
    'earnings beat': 2.0,
    'rate cut': 1.5,
    'rate cuts': 1.5,
    'strong earnings': 2.0,
    'profit surge': 2.2,
    'low unemployment': 1.5,
    'low inflation': 1.5,
    'price target raised': 2.0,
    'upgrade to buy': 2.2,
    'bull market': 2.0,
    'short squeeze': 1.8,
    'partnership announced': 1.2,
    
    'all-time low': -2.5,
    'record low': -2.5,
    'missed expectations': -2.0,
    'earnings miss': -2.0,
    'rate hike': -1.5,
    'rate hikes': -1.5,
    'high inflation': -2.0,
    'high unemployment': -2.0,
    'bank run': -3.0,
    'liquidation': -2.2,
    'sec lawsuit': -2.5,
    'investigation launched': -2.0,
    'bear market': -2.0,
    'downgrade to sell': -2.2,
    'debt default': -3.0,
    'layoffs announced': -1.8,
    'bankruptcy filing': -3.0
}

# Single-word financial valence weights
VALENCE_DICT = {
    # Strong Positive (+2.0 to +3.0)
    'surge': 2.2, 'surging': 2.2, 'skyrocket': 2.8, 'soar': 2.5, 'soaring': 2.5,
    'breakout': 2.0, 'outperform': 2.2, 'boom': 2.0, 'rally': 2.2, 'rallying': 2.2,
    # Moderate Positive (+1.0 to +1.8)
    'gain': 1.2, 'gains': 1.2, 'gaining': 1.2, 'bullish': 1.8, 'profit': 1.5,
    'profitable': 1.5, 'grow': 1.2, 'growth': 1.4, 'growing': 1.2, 'rise': 1.2,
    'rising': 1.2, 'upgrade': 1.8, 'upgraded': 1.8, 'recovery': 1.4, 'rebound': 1.6,
    'optimism': 1.4, 'optimistic': 1.5, 'boost': 1.3, 'boosted': 1.3, 'dividend': 1.0,
    'accumulate': 1.2, 'buy': 1.0,
    
    # Strong Negative (-2.0 to -3.0)
    'crash': -2.8, 'crashing': -2.8, 'plunge': -2.5, 'plunging': -2.5, 'plummet': -2.6,
    'collapse': -2.8, 'bankrupt': -3.0, 'bankruptcy': -3.0, 'fraud': -3.0, 'scam': -3.0,
    'liquidate': -2.4, 'bloodbath': -2.8, 'selloff': -2.2,
    # Moderate Negative (-1.0 to -1.8)
    'loss': -1.5, 'losses': -1.5, 'drop': -1.3, 'dropping': -1.3, 'dropped': -1.3,
    'decline': -1.3, 'declining': -1.3, 'bearish': -1.8, 'slump': -1.6, 'sink': -1.5,
    'sinking': -1.5, 'downgrade': -1.8, 'downgraded': -1.8, 'warning': -1.5, 'warn': -1.4,
    'risk': -1.0, 'risky': -1.2, 'recession': -2.0, 'fear': -1.6, 'sell': -1.0,
    'panic': -2.2, 'deficit': -1.5, 'default': -2.4, 'lawsuit': -1.8
}

INTENSIFIERS = {
    'substantially': 1.5,
    'significantly': 1.5,
    'massively': 1.8,
    'sharply': 1.6,
    'heavily': 1.4,
    'slightly': 0.6,
    'marginally': 0.5,
    'barely': 0.4
}

NEGATIONS = {
    'not', 'no', 'never', 'none', 'neither', 'nor', 'hardly', 'scarcely',
    "wasn't", "weren't", "isn't", "aren't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "cannot", "can't", "couldn't"
}


def analyze_text_sentiment(text: str) -> float:
    """
    Performs context-aware financial sentiment scoring on text.
    Returns a normalized compound score in range [-1.0, 1.0].
    """
    if not text or not text.strip():
        return 0.0

    lower_text = text.lower()
    total_score = 0.0
    matched_phrases = set()

    # 1. Match multi-word financial phrases first
    for phrase, weight in FINANCIAL_PHRASES.items():
        if phrase in lower_text:
            total_score += weight
            matched_phrases.add(phrase)

    # 2. Token-level analysis with negation and intensifiers
    cleaned = re.sub(r'[^\w\s]', ' ', lower_text)
    words = cleaned.split()

    for i, word in enumerate(words):
        # Skip if word is part of an already matched phrase
        if any(word in p for p in matched_phrases):
            continue

        if word in VALENCE_DICT:
            val = VALENCE_DICT[word]
            
            # Check window of up to 3 prior words for negation and intensifiers
            window = words[max(0, i - 3):i]
            is_negated = any(w in NEGATIONS for w in window)
            
            multiplier = 1.0
            for w in window:
                if w in INTENSIFIERS:
                    multiplier = INTENSIFIERS[w]
                    break

            val *= multiplier
            if is_negated:
                val = -0.75 * val  # Reverse sentiment with standard dampening

            total_score += val

    # Normalize score into [-1.0, 1.0] using standard soft bounding
    # score / sqrt(score^2 + 15)
    if total_score == 0.0:
        return 0.0
    normalized = total_score / ( (total_score ** 2 + 15.0) ** 0.5 )
    return float(round(max(-1.0, min(1.0, normalized)), 3))


def get_news_sentiment(symbol: str, is_crypto: bool = False) -> dict:
    """
    Fetches latest news headlines via Google News RSS with 15-minute TTL caching.
    Returns structured sentiment metrics including score, label, impact, and headlines.
    """
    symbol = symbol.upper()
    cache_key = f"sentiment_{symbol}_{is_crypto}"
    cached = sentiment_cache.get(cache_key)
    if cached is not None:
        return cached

    # Query construction
    if symbol in ('XAUUSD', 'GOLD', 'GC=F'):
        query = "gold commodity price inflation market"
    elif is_crypto:
        clean = symbol.replace('-USD', '').replace('USDT', '')
        query = f"{clean} cryptocurrency market price"
    else:
        query = f"{symbol} stock market earnings forecast"

    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    headlines = []
    total_sentiment = 0.0

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
        }
        response = requests.get(rss_url, headers=headers, timeout=6)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = root.findall('.//item')[:8]

            for item in items:
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                source = item.find('source').text if item.find('source') is not None else "News Source"

                # Clean publisher suffix (e.g., " - Bloomberg")
                title_clean = re.sub(r'\s+-\s+[^(-]+$', '', title).strip()
                sent = analyze_text_sentiment(title_clean)
                total_sentiment += sent

                headlines.append({
                    "title": title_clean,
                    "link": link,
                    "date": pub_date,
                    "source": source,
                    "sentiment": sent
                })

    except Exception as e:
        print(f"Error fetching sentiment for {symbol}: {e}")

    count = len(headlines)
    score = float(round(total_sentiment / count, 3)) if count > 0 else 0.0

    # Categorize label
    if score >= 0.35:
        label = "Strongly Bullish"
        impact_type = "positive"
        impact_dir = "Strong Bullish Catalyst"
        effect = "High positive news volume suggests upward momentum."
    elif score >= 0.10:
        label = "Bullish"
        impact_type = "positive"
        impact_dir = "Moderate Bullish Bias"
        effect = "Favorable news headlines indicate mild positive sentiment."
    elif score <= -0.35:
        label = "Strongly Bearish"
        impact_type = "negative"
        impact_dir = "Strong Bearish Catalyst"
        effect = "Significant negative news coverage indicates elevated downside risk."
    elif score <= -0.10:
        label = "Bearish"
        impact_type = "negative"
        impact_dir = "Moderate Bearish Bias"
        effect = "Cautious headlines indicate mild headwinds."
    else:
        label = "Neutral"
        impact_type = "neutral"
        impact_dir = "Balanced Impact"
        effect = "Balanced market sentiment indicates no immediate headline-driven volatility."

    result = {
        "score": score,
        "label": label,
        "headlines": headlines,
        "impact_type": impact_type,
        "impact_direction": impact_dir,
        "predicted_effect": effect,
        "next_move_pct": float(round(score * 1.5, 2)),
        "max_impact_pct": float(round(score * 3.0, 2))
    }

    # Cache for 15 minutes
    sentiment_cache.set(cache_key, result, ttl=900)
    return result
