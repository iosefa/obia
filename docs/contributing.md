# Contributing

## Development setup

```bash
git clone https://github.com/iosefa/obia.git
cd obia
conda env create -f environment.yml
conda activate obia
```

## Documentation

```bash
mkdocs serve
```

## Publishing releases

PyPI uses Trusted Publishing. In the `obia` project's PyPI publishing settings,
configure a GitHub publisher with these exact values:

- Owner: `iosefa`
- Repository: `obia`
- Workflow filename: `publish-pypi.yml`
- Environment: `pypi`

Set the version in `pyproject.toml`, merge the release commit, and push the matching
`vX.Y.Z` tag. The publishing workflow builds and checks the distributions before
uploading them to PyPI.

To retry an unpublished release after fixing a publishing failure, run the
workflow from `main` with the existing tag:

```bash
gh workflow run publish-pypi.yml --ref main -f tag=v0.1.4
```

The manual run builds the tagged source and verifies that its package version
matches the tag. It does not move or recreate the tag. PyPI does not allow
overwriting an already published distribution.

## Contribution focus

Current high-value contributions:

- tests for `obia/segmentation`, `obia/detection`, and `obia/classification`
- reproducible examples using small public datasets
- API cleanup and typing improvements
- docs improvements that match actual module behavior
