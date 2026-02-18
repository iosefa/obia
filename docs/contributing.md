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

## Contribution focus

Current high-value contributions:

- tests for `obia/segmentation`, `obia/detection`, and `obia/classification`
- reproducible examples using small public datasets
- API cleanup and typing improvements
- docs improvements that match actual module behavior
