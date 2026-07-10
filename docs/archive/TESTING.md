# Testing Guide

## Overview

AlphaSignal has a comprehensive test suite with 152 tests covering all major components. Test coverage is **92%**.

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=alphasignal --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see detailed coverage.

### Run Specific Test Files

```bash
# Test retrieval system
pytest alphasignal/tests/test_retriever.py

# Test API endpoints
pytest alphasignal/tests/test_api.py

# Test ingestion
pytest alphasignal/tests/test_edgar.py
pytest alphasignal/tests/test_news.py
```

### Run Specific Tests

```bash
pytest alphasignal/tests/test_api.py::test_query_endpoint_returns_200
```

### Run Tests with Verbose Output

```bash
pytest -v
```

### Run Tests and Stop on First Failure

```bash
pytest -x
```

## Test Structure

```
alphasignal/tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_api.py              # API integration tests (9 tests)
├── test_chunker.py          # Chunking tests (9 tests)
├── test_edgar.py            # EDGAR ingestion tests (7 tests)
├── test_evaluator.py        # Evaluation metrics tests (7 tests)
├── test_generation.py       # RAG generation tests (7 tests)
├── test_health.py           # Health endpoint tests (4 tests)
├── test_news.py             # News ingestion tests (8 tests)
├── test_retriever.py        # Hybrid retrieval tests (10 tests)
├── test_sentiment.py        # Sentiment extraction tests (7 tests)
└── test_store.py            # Vector/metadata store tests (7 tests)
```

## Test Categories

### Unit Tests

Test individual components in isolation:

- `test_chunker.py` - Semantic chunking logic
- `test_store.py` - FAISS and SQLite operations
- `test_evaluator.py` - Retrieval metrics (MRR, NDCG, Hit@k)
- `test_generation.py` - RAG answer synthesis
- `test_sentiment.py` - Sentiment extraction

### Integration Tests

Test multiple components together:

- `test_api.py` - Full API endpoints with mocked dependencies
- `test_retriever.py` - Hybrid retrieval (BM25 + FAISS + reranking)
- `test_edgar.py` - EDGAR fetching and parsing
- `test_news.py` - RSS feed parsing

### Mocking Strategy

Tests use `unittest.mock.patch` to mock external dependencies:

- **OpenAI API**: Mocked in all tests to avoid API calls and costs
- **SEC EDGAR Downloader**: Mocked to avoid network requests
- **RSS Feeds**: Mocked with synthetic feed entries
- **File I/O**: Uses temporary directories via `pytest.fixture(tmp_data_dir)`

## Coverage by Module

| Module | Coverage |
|--------|----------|
| Core retrieval | 92% |
| Generation | 100% |
| Stores | 94% |
| API routes | 83-86% |
| Ingestion | 77-87% |
| Overall | **92%** |

## Writing New Tests

### Test File Template

```python
"""Tests for [module name]."""

import pytest
from alphasignal.[module] import [Component]


@pytest.fixture
def component():
    """Create test component."""
    return Component()


def test_component_does_something(component):
    """Test that component does something."""
    result = component.do_something()
    assert result == expected
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
def test_with_fixtures(client, mock_config, tmp_data_dir):
    """Test using multiple fixtures."""
    # client: FastAPI TestClient
    # mock_config: Test configuration dict
    # tmp_data_dir: Temporary directory Path
    pass
```

### Mocking External APIs

```python
from unittest.mock import patch, MagicMock

def test_with_mock():
    """Test with mocked external API."""
    with patch('alphasignal.embeddings.embedder.OpenAI') as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )

        # Test code here
```

## Continuous Integration

To run tests in CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=alphasignal --cov-report=xml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting Tests

### Tests Fail with "OpenAI API key not found"

**Solution:** The client fixture in `conftest.py` sets a dummy API key. Make sure you're using the `client` fixture, not creating a TestClient directly.

### Tests Fail with "No module named 'alphasignal'"

**Solution:** Run tests from the project root directory:
```bash
cd /path/to/AlphaSignal
pytest
```

### Tests Are Slow

**Solution:** Tests that hit real APIs or files are mocked. If tests are slow:
1. Check that mocks are properly configured
2. Use `pytest -x` to stop on first failure
3. Run specific test files instead of all tests

### FAISS/SQLite Tests Fail

**Solution:** Tests use temporary directories. Make sure `/tmp` is writable and has sufficient space.

## Test Maintenance

- **Add tests** for all new features and bug fixes
- **Update mocks** when external APIs change
- **Keep coverage above 85%** - run `pytest --cov` regularly
- **Review failing tests** immediately - don't let them accumulate

## Best Practices

1. **One assertion per test**: Makes failures easy to debug
2. **Descriptive test names**: Use `test_component_does_x_when_y`
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Mock external dependencies**: Keep tests fast and isolated
5. **Use fixtures**: Reduce code duplication
6. **Test edge cases**: Not just the happy path
