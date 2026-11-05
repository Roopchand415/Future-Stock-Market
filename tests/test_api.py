import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    res = client.get('/health')
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data['status'] == 'healthy'
    assert 'academic' in json_data['version']


def test_price_not_found(client):
    res = client.get('/api/price/INVALID_TICKER_XYZ_999')
    # Should return either 404 or gracefully handled error
    assert res.status_code in (404, 500)
