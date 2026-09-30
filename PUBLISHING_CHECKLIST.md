# Publishing checklist

Checklist for source releases. Completed publication metadata is recorded below;
licensing and corpus redistribution choices remain the maintainer's responsibility.

- [x] Set repository URLs in the changelog, package metadata, citation, and badges.
- [ ] Replace the generic contributor entry in `CITATION.cff` with the preferred
  author name, affiliation, ORCID, and repository URL.
- [ ] Confirm Apache-2.0 is the intended code license.
- [ ] Review `THIRD_PARTY_MODELS.md` and `THIRD_PARTY_NOTICES.md`; obtain any
  commercial PyMuPDF/MuPDF license required by the intended distribution model.
- [ ] Confirm every shared catalog is owned, licensed, or redistribution-cleared.
- [ ] Keep source PDFs, extracted text, embeddings, model weights, credentials,
  databases, and run artifacts out of Git.
- [ ] Run `python scripts/build_source_release.py` and upload only the generated
  allowlisted ZIP or an equivalently reviewed Git tree.
- [ ] Run the clean-environment commands in `docs/REPRODUCIBILITY.md`.
- [ ] Scan the staged tree and Git history with a secret scanner before publishing.
- [ ] Create a release tag matching `src/industrial_catalog/__init__.py` and update
  the changelog comparison links.

The checked-in corpus statistics and aggregate measurements are research evidence,
not a redistribution grant for the underlying PDFs or model responses.
