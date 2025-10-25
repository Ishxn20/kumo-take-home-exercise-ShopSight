# ShopSight

## Thought Process

- I prioritised the past-sales flow (search → KPIs → weekly momentum) because it can be powered end-to-end by the real H&M dataset, satisfying the brief’s requirement for one grounded journey.
- The forecast tile, buyer personas, and recommendation cards are deterministic mock-ups so I could keep the scope tight while still showcasing the UI surface area.
- A local warehouse keeps data access snappy, and Streamlit made it easy to combine analytics, mocks, and LLM narration quickly.
- Narrative, action plan, trend commentary, and the chat assistant call OpenAI when a key is provided, but every surface degrades to deterministic copy so the prototype works offline.
- Shared settings (data paths, top-N sample, UI placeholders, help links) live in `config.py`, reducing the number of places reviewers need to touch.

## Architecture

```
app/
  data_loader.py        # SQLite accessors for catalog search & daily metrics
  insights.py           # KPI calculations, trend aggregation, mocked forecast/segments/actions
  llm.py                # Narrative, action plan, trend commentary, chat assistant
  streamlit_app.py      # Streamlit UI
data/
  shopsight.db          # SQLite warehouse built from parquet
  transactions/         # Parquet shards from s3://kumo-public-datasets/…
  articles.parquet      # Article metadata parquet (ignored)
scripts/
  load_hm_data.py       # Parquet → SQLite (keeps top-N articles)
tests/
  test_insights.py      # Smoke tests for KPIs, trends, mocked forecast
config.py               # Shared configuration (paths, top-N, UI defaults)
```

---

## Data Strategy

1. **Download the source parquet files**
   ```bash
   aws s3 sync s3://kumo-public-datasets/hm_with_images/transactions/ data/transactions/ --no-sign-request
   aws s3 cp s3://kumo-public-datasets/hm_with_images/articles/part-00000-...parquet data/articles.parquet --no-sign-request
   ```
2. **Build the local warehouse**
   ```bash
   python scripts/load_hm_data.py --top-n 60
   ```
   The loader scans all transaction shards, ranks articles by revenue, keeps the 60 best sellers, and writes their daily metrics plus metadata into `data/shopsight.db`. Historical KPIs and the Past Sales chart read straight from this database.
3. **Mock the forward-looking insights**
   - `generate_mock_forecast(product_name)` → reproducible forecast numbers keyed off the product name.
   - `generate_mock_segments(product_name)` → persona placeholders (Digital Loyalists / Store Stylists / Seasonal Gifters).
   - `build_mock_additional_insights(...)` → copy-only recommendations.

---

## To Run the Demo

```bash
# 1. Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the UI
streamlit run app/streamlit_app.py
```

**LLM setup**
```bash
export OPENAI_API_KEY=sk-...
# optional override
export SHOP_SIGHT_OPENAI_MODEL=gpt-4o-mini
```

Without a key everything still works; the narrative, actions, and chat fall back to a deterministic output.

## Configuration

- `config.py` centralises repository settings: data paths, the default top-N article sample, UI placeholders, chat history limits, and help/feedback links.
- `scripts/load_hm_data.py`, `app/data_loader.py`, and `app/streamlit_app.py` import these constants so behaviour can be tweaked from one file.

## LLM Integration

`app/llm.py` powers the agentic layer:
- Narrative summary
- Recommended action plan
- Trend commentary
- Conversational assistant

All prompts instruct the model to return JSON so the UI can render structured text safely. If JSON parsing fails or the call errors, the code falls back to deterministic copy built from the real KPIs.

**How it behaves**
- With `OPENAI_API_KEY` set, the app calls OpenAI’s Responses API (`gpt-4o-mini` by default, override via `SHOP_SIGHT_OPENAI_MODEL`).
- Without a key—or if the API returns an error—the same helpers fall back to deterministic copy that plugs in the real historical metrics, so the dashboard stays fully functional offline.

---

## 🔍 Real vs. Mocked

| Area                               | Status | Details |
|------------------------------------|--------|---------|
| Product catalog + search           | Real | SQL queries over `article_summary` in SQLite |
| KPIs & weekly sales chart          | Real | Aggregations on `article_daily_metrics` |
| Channel / region mix charts        | Real | Derived from daily metrics (regions hashed from customer_id) |
| Forecast tile                      | Mocked | Deterministic projection keyed by product name |
| Customer segments                  | Mocked | Persona placeholders highlighting potential UX |
| Recommendation cards               | Mocked | Seller-facing guidance seeded by product name |
| Narrative / actions / chat         | Optional | Live when OpenAI key present; deterministic fallback otherwise |

---

## Gaps & Next Steps

1. **Real forecasting.** Swap the mocked projection for Prophet/ARIMA/Kumo models, backtest on holdout months, and surface a confidence band derived from residuals.
2. **Data-driven personas.** Join `customers.parquet`, calculate RFM-style clusters, and feed real segment descriptions + recommended actions.
3. **Agent actions.** Allow the chat assistant to execute SQL or forecast functions instead of just describing them.
4. **Comparison & benchmarking.** Enable multi-product views or baselines (“compare against 706016002”).
5. **Deployability.** Add authentication, caching, CI, and packaging for Streamlit Cloud or an internal environment.

These items are called out in the code and README so the next iteration is straightforward.

---

## Tests

Smoke tests verify that the historical analytics remain intact:

```bash
pytest
```

---
