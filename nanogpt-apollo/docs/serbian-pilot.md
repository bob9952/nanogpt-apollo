# Serbian manual pilot

Use the Ubuntu server application directory in PuTTY. Copy the following TXT
files from `nanogpt-apollo/docs/` on Windows into `data/inputs/` via WinSCP.
The four extracts are already UTF-8.

- `cela dokumentacija.txt`
- `UputstvoPostaCG.txt`
- `ApolloG4 Uputstvo - Lovcen Auto OFFLINE Jul 2010.txt`
- `G5 Igorl.txt`

Keep the two English TXT manuals from the previous run in the same directory.
The G5 manual and the installation-specific guides provide evidence for their
own versions and processes. Check which manual an answer cites before applying
its instructions to another installation.

## Prepare and check retrieval

```bash
mkdir -p data/inputs data/private .cache/huggingface
ls -lh data/inputs
```

After all six files are present and this app revision is on Ubuntu, run:

```bash
docker compose build assistant

APOLLO_CORPUS_DIR="$PWD/data/private" docker compose run --rm \
  -v "$PWD/data/inputs:/inputs:ro" \
  -v "$PWD/data/private:/prepared" \
  -v "$PWD/.cache/huggingface:/tokenizer-cache" \
  assistant prepare \
  --manual '/inputs/Apollo G4 ERP Users Guide v2.txt' \
  --manual '/inputs/specification of app english.txt' \
  --manual '/inputs/cela dokumentacija.txt' \
  --manual '/inputs/UputstvoPostaCG.txt' \
  --manual '/inputs/ApolloG4 Uputstvo - Lovcen Auto OFFLINE Jul 2010.txt' \
  --manual '/inputs/G5 Igorl.txt' \
  --output /prepared/corpus-v5 \
  --tokenizer-cache /tokenizer-cache
```

`prepare` publishes a new bundle. The temporary corpus mount allows preparation
before `corpus-v5` exists. After preparation succeeds, start the services if
needed and use the new bundle and collection for the rest of the pilot.

```bash
docker start ollama
docker compose up -d chroma
export APOLLO_CORPUS_DIR="$PWD/data/private/corpus-v5"
docker compose run --rm assistant doctor
docker compose run --rm assistant index --corpus /corpus --collection apollo-manuals-v5 --kind manual

question='Gde se podešava konto zaliha za artikal i gde se proverava podrazumevano konto za magacin?'
docker compose run --rm assistant search "$question" \
  --corpus /corpus --collection apollo-manuals-v5 --kind manual --top-k 10
```

Look for the warehouse/item-type defaults and item override in the English G4
guide, lines 443–459. The Serbian `Konta zaliha` section in `cela dokumentacija.txt`
around line 5517 is part of **fixed assets**. If the top ten miss the G4 steps,
repeat `search` with `LIN001M001 LIT002M001` to check whether the passages are in
the collection. A hit by form code and a miss by natural language point to
ranking. For a Serbian positive control, search `Koje opcije LOTCTRL postoje za
artikal u LIT002M001?`. Check `cela dokumentacija.txt` around lines 2357–2359
for batch, serial number, and no tracking.

## Compare local answer models

Use a question whose top three passages contain the relevant evidence. Try
Qwen 2B Q4 first against the installed Qwen 4B. The 2B Q4 tag is a roughly
1.9 GB download. Both runs use the same E5 vectors, question, and generation
settings. [Ollama Qwen 3.5 tags](https://ollama.com/library/qwen3.5/tags)

```bash
docker exec ollama ollama pull qwen3.5:2b-q4_K_M
docker compose run --rm assistant doctor --model qwen3.5:2b-q4_K_M

for model in qwen3.5:4b-q4_K_M qwen3.5:2b-q4_K_M; do
  echo "$model"
  for run in 1 2; do
    docker compose run --rm assistant ask "$question" --model "$model" \
      --corpus /corpus --collection apollo-manuals-v5 --kind manual || break 2
  done
  docker exec ollama ollama ps
  docker stats --no-stream ollama
  docker exec ollama ollama stop "$model"
done
```

Compare the second runs with the models loaded. Record the source IDs, whether
the answer and citations are correct, Serbian Latin, time to first text, total
time, tokens per second, and the container memory snapshot. Use the original
item-56 question as a follow-up. The answer should ask for the missing live
configuration and report the applicable documented checks.

Once retrieval works, use the local results to decide whether a GPU experiment
is worth spending Modal credits on. Fine-tuning would need reviewed examples of
a specific behavior to improve. The manuals belong in the local retrieval corpus.
