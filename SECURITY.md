# Security policy

## Supported version

The current `0.1.x` research line receives security fixes. This project is alpha
software and has not undergone an independent security audit.

## Reporting a vulnerability

Do not open a public issue containing credentials, private catalog content, path
disclosure, or a working exploit. Use GitHub's private vulnerability reporting from
the repository **Security** tab. If that feature is unavailable, contact the
maintainer through the private contact method shown on the repository owner's
profile. Include the affected version, impact, reproduction steps, and a minimal
synthetic fixture when possible.

## Deployment guidance

- Bind model endpoints to loopback or a protected internal network by default.
- Use authentication and TLS when endpoints cross a trust boundary.
- Store API keys in environment variables or a secret manager, never in YAML.
- Treat PDFs as untrusted input; isolate processing and keep dependencies patched.
- Do not render or execute embedded PDF attachments or external links.
- Keep catalog data and model caches outside the source checkout.
- Limit service accounts to the corpus and output directories they require.
- Inspect logs before sharing because raw model payloads may contain source text.
- Do not expose SQLite databases or checkpoints as public build artifacts.

## Threat model boundary

The code validates data lineage and output schemas, but it is not a malware scanner,
data-loss-prevention product, or safety certification system. Model output can be
incorrect even when well formed. Preserve citations and require domain review for
safety-critical or regulated decisions.
