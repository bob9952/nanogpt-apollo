# Serbia Ministry of Finance opinion pilot

Start with three opinions from 2025 about PDV and electronic invoices. The
export keeps the Serbian source text, issuer, opinion number, date, original PDF
link and page range. The Corpus v5 manuals and their Chroma collection remain
available for Apollo product questions.

- `000501114` — input VAT deduction using an electronic invoice
- `001855973` — conditions for input VAT deduction
- `003002091` — a paper invoice in a scenario involving electronic invoicing

These documents record dated Ministry interpretations. A current-law answer
needs the official provision applicable on the relevant transaction date.
For this first run, test retrieval and verify the opinion against the original
Ministry PDF. The pages printed by `search` refer to the exported text. Read the
export header for the source PDF page range.

## Run on Ubuntu

After the submodule and app changes are available in your server checkout, run
from the **parent Git repository** and then enter the app directory:

```bash
cd ~/ai/nanogpt-apollo
git submodule update --init -- serbia-law
git submodule status -- serbia-law
cd nanogpt-apollo
```

The pinned checkout includes the public opinion source files. Docker mounts
them read-only. The exporter and corpus write into the app's ignored
`data/private/` directory.

```bash
mkdir -p data/private .cache/huggingface
export APOLLO_CORPUS_DIR="$PWD/data/private"
docker compose build assistant

docker compose run --rm \
  -v "$PWD/../serbia-law:/serbia-law:ro" \
  -v "$PWD/data/private:/prepared" \
  --entrypoint python assistant -m apollo_rag.opinions \
  --repository /serbia-law \
  --output /prepared/finance-opinions-inputs-v1

docker compose run --rm \
  -v "$PWD/data/private:/prepared" \
  -v "$PWD/.cache/huggingface:/tokenizer-cache" \
  assistant prepare \
  --manual /prepared/finance-opinions-inputs-v1/mfin-000501114-2025-10520-004-000-011-004.txt \
  --manual /prepared/finance-opinions-inputs-v1/mfin-001855973-2025-10520-004-000-011-004.txt \
  --manual /prepared/finance-opinions-inputs-v1/mfin-003002091-2025-10520-004-000-011-004.txt \
  --output /prepared/finance-opinions-v1 \
  --tokenizer-cache /tokenizer-cache
```

Once preparation reports three documents, index and try a Latin-script query:

```bash
docker compose up -d chroma
export APOLLO_CORPUS_DIR="$PWD/data/private/finance-opinions-v1"
docker compose run --rm assistant index \
  --corpus /corpus --collection serbia-finance-opinions-v1 --kind manual
docker compose run --rm assistant search \
  'Mišljenje o papirnom računu umesto elektronske fakture i odbitku prethodnog PDV' \
  --corpus /corpus --collection serbia-finance-opinions-v1 --kind manual --top-k 5
```

Look for the opinion with record ID beginning `003002091` and confirm the date
`2025-09-15` in the export. Check whether the retrieved text covers the actual
scenario. Separately try a question about deduction based on an electronic
invoice and look for `000501114`. Compare the passage against the original PDF
before testing generated answers.

To return to ERP manual queries, set
`APOLLO_CORPUS_DIR="$PWD/data/private/corpus-v5"` and use `apollo-manuals-v5`.
