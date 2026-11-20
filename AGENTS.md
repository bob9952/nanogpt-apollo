# Apollo application workflow

These instructions apply to work in `nanogpt-apollo/`.

## Apollo development

- Use the checked-out application code for local edits and inspection. Run the application, corpus preparation, Docker Compose, and runtime checks on Ubuntu through the user's PuTTY session. Provide focused server commands and use their output as evidence.
- Keep private manuals, SVN exports, and generated corpus and index data out of Git. Preserve the upstream submodules. Use the existing Chroma and Ollama workflow.
- Run the nearest relevant existing check. Add a regression test only when it protects meaningful behavior or captures a reproduced bug, or when required. Report which checks are still pending.
- For retrieval changes, inspect the passages before assessing generated answers. Keep the assistant's answers in Serbian Latin and distinguish documented guidance from unknown live ERP configuration.
