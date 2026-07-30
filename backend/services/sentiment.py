import urllib.parse
import xml.etree.ElementTree as ET
import requests
import re

# Financial-focused sentiment lexicon
POSITIVE_WORDS = {
    'bull', 'bullish', 'surge', 'growth', 'grow', 'gain', 'profit', 'rise', 'rising', 'upward', 
    'success', 'successful', 'beat', 'outperform', 'upgrade', 'buy', 'strong', 'positive', 
    'jump', 'soar', 'highest', 'rally', 'win', 'winning', 'climb', 'higher', 'expand', 'expansion',
    'optimistic', 'confidence', 'green', 'boost', 'recovery', 'recover', 'high', 'breakout', 'hype'
}

# Negation words to handle simple sentiment flips (e.g. "not bullish")
NEGATIONS = {'not', 'no', 'never', 'none', 'neither', 'nor', 'barely', 'hardly'}

NEGATIVE_WORDS = {
    'bear', 'bearish', 'drop', 'fall', 'falling', 'loss', 'lose', 'decline', 'downward', 
    'fail', 'failed', 'failure', 'underperform', 'downgrade', 'sell', 'weak', 'negative', 
    'plummet', 'crash', 'lowest', 'slump', 'sink', 'concern', 'risk', 'risky', 'slide', 
    'red', 'shrink', 'inflation', 'pressure', 'warn', 'warning', 'worry', 'worried', 'debt', 'low',
    'lawsuit', 'dump', 'panic'
}

def analyze_text_sentiment(text):
    """
    Performs a lexicon-based sentiment analysis on a string.
    Returns a score between -1.0 (strongly negative) and 1.0 (strongly positive).
    """
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()
    
    pos_count = 0
    neg_count = 0
    
    for i, word in enumerate(words):
        is_negated = False
        # Look back up to 2 words for negations
        for j in range(max(0, i-2), i):
            if words[j] in NEGATIONS:
                is_negated = not is_negated
                
        if word in POSITIVE_WORDS:
            if is_negated:
                neg_count += 1
            else:
                pos_count += 1
        elif word in NEGATIVE_WORDS:
            if is_negated:
                pos_count += 1
            else:
                neg_count += 1
                
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total

def get_news_sentiment(symbol, is_crypto=False):
    """
    Fetches the latest news headlines from Google News RSS feed for the asset,
    analyzes sentiment for each headline, and aggregates the overall sentiment.
    """
    symbol = symbol.upper()
    
    # 1. Build Query
    if symbol in ('XAUUSD', 'GOLD'):
        query = "gold price commodity market"
    elif is_crypto:
        query = f"{symbol} cryptocurrency crypto market news"
    else:
        query = f"{symbol} stock market earnings news"
        
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    headlines = []
    overall_score = 0.0
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(rss_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = root.findall('.//item')[:6] # Grab latest 6 news items
            
            total_sentiment = 0.0
            sentiment_items_count = 0
            
            for item in items:
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                source = item.find('source').text if item.find('source') is not None else "News Source"
                
                # Strip off source suffix at the end of Google News titles (e.g., " - Reuters")
                title_clean = re.sub(r'\s+-\s+[^(-]+$', '', title).strip()
                
                sentiment = analyze_text_sentiment(title_clean)
                
                total_sentiment += sentiment
                sentiment_items_count += 1
                
                headlines.append({
                    "title": title_clean,
                    "link": link,
                    "date": pub_date,
                    "source": source,
                    "sentiment": sentiment
                })
                
            if sentiment_items_count > 0:
                overall_score = total_sentiment / sentiment_items_count
                
    except Exception as e:
        print(f"Error fetching news sentiment for {symbol}: {e}")
        
    # Determine Sentiment Label, color/score indicator, and next-move impact details
    if overall_score >= 0.35:
        label = "Strongly Bullish"
        impact_direction = "Bullish Boost"
        predicted_effect = "Strong positive news coverage is expected to accelerate the next upward move."
        impact_type = "positive"
    elif overall_score >= 0.05:
        label = "Bullish"
        impact_direction = "Bullish Support"
        predicted_effect = "Optimistic news sentiment is providing mild upward support for the next move."
        impact_type = "positive"
    elif overall_score <= -0.35:
        label = "Strongly Bearish"
        impact_direction = "Bearish Drag"
        predicted_effect = "Strong pessimistic sentiment is likely to pull the next move significantly downward."
        impact_type = "negative"
    elif overall_score <= -0.05:
        label = "Bearish"
        impact_direction = "Bearish Resistance"
        predicted_effect = "Negative news coverage is introducing downward drag/resistance for the next move."
        impact_type = "negative"
    else:
        label = "Neutral"
        impact_direction = "No Impact"
        predicted_effect = "Balanced news sentiment indicates neutral impact, keeping the next move in line with technical indicators."
        impact_type = "neutral"

    # Calculate estimated next-move percentage impact (Day 1 impact in 7-day forecast)
    # The bias factor over 7 days is 0.05 * overall_score.
    # On Day 1 (first next move), the impact is 1/7th of that.
    next_move_pct = round((0.05 * overall_score * (1.0 / 7.0)) * 100, 3)
    max_impact_pct = round((0.05 * overall_score) * 100, 2)
        
    return {
        "score": round(overall_score, 2), # Between -1.0 and 1.0
        "label": label,
        "headlines": headlines,
        "impact_direction": impact_direction,
        "predicted_effect": predicted_effect,
        "impact_type": impact_type,
        "next_move_pct": next_move_pct,
        "max_impact_pct": max_impact_pct
    }
