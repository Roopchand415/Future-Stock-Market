import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from services.sentiment import analyze_text_sentiment


def test_positive_financial_sentiment():
    score = analyze_text_sentiment("Company reports record high quarterly profits and strong revenue surge")
    assert score > 0.30, f"Expected positive score, got {score}"


def test_negative_financial_sentiment():
    score = analyze_text_sentiment("Severe debt default warning as stock plummets following lawsuit investigation")
    assert score < -0.30, f"Expected negative score, got {score}"


def test_negation_inversion():
    pos_score = analyze_text_sentiment("The market outlook is bullish")
    neg_score = analyze_text_sentiment("The market outlook is not bullish")
    assert pos_score > 0
    assert neg_score < 0, f"Negation was not detected! Got {neg_score}"


def test_contextual_phrases():
    # 'low unemployment' should be positive, not penalized for the word 'low'
    score = analyze_text_sentiment("Economy exhibits low unemployment and healthy growth")
    assert score > 0.20, f"Expected positive score for low unemployment, got {score}"
