# Contributing

Thank you for considering contributing to this project.

## Code of conduct

This project is governed by a respectful, professional, and inclusive standard. Please be kind, constructive, and focused on the technical problem.

## Development setup

1. Clone the repository.
2. Create a virtual environment with `uv`.
3. Install dependencies:

```bash
uv sync --group dev
```

4. Copy the environment template:

```bash
cp .env.example .env
```

5. Fill in the required values:
   - `GROQ_API_KEY` or another LLM provider key
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_BASE_URL`
   - `APP_ENV`, `LOG_LEVEL`, `RATE_LIMIT`, and cache settings

## Local services required

This project expects one of the following runtime setups:

- a local Langfuse instance, or a hosted Langfuse deployment
- a PostgreSQL instance for app data or local Docker stack
- Ollama running locally, or a supported LLM provider such as Groq
- valid environment variables in `.env`

## Running locally

```bash
uv run python -m uvicorn app.main:app --reload
```

## Testing

Run the suite with:

```bash
uv run pytest --cov=app --cov-report=term-missing tests -q
```

## Commit message conventions

Use conventional commits:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation updates
- `chore:` for maintenance work
- `test:` for test additions or updates
- `refactor:` for code cleanup that does not change behavior

Examples:

```bash
git commit -m "feat(api): add cache status endpoint"
git commit -m "fix(security): mask PII before validation"
git commit -m "docs(readme): add local startup instructions"
```

## Pull request process

1. Create a branch from `main`.
2. Keep changes focused and small.
3. Add or update tests for functional changes.
4. Run the relevant test suite locally.
5. Open a pull request with a clear summary and testing notes.

## Review expectations

- keep the codebase consistent with the existing style
- avoid introducing outdated library patterns
- do not rely on live external services during automated tests
- document any env or runtime assumptions clearly
