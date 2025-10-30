# pesto-reduction

A Python package for astronomical image reduction and mosaic creation, specifically designed for processing telescope observation data.

## Overview

`pesto-reduction` provides tools to process raw astronomical images through calibration, stacking, and mosaic generation. The package handles the complete reduction pipeline from flat-field correction to final mosaic assembly.

## Installation

```bash
pip install -e .
```

This will install the package and make the following command-line tools available:
- `pesto-create-flat`
- `pesto-create-stack`
- `pesto-mosaic`

## Requirements

- Python >= 3.9
- Astrometry.net index files (see [Setup](#astrometry-index-files) below)

## Usage

The typical workflow consists of three main steps:

### 1. Create Master Flat

Generate a master flat-field calibration frame from dome flat exposures:

```bash
pesto-create-flat \
    --data_dir /path/to/flat/frames \
    --output_path /path/to/output/master_flat.fits \
    --bias_level 300 \
    --crop_size 500
```

**Parameters:**
- `--data_dir`: Directory containing raw flat-field frames
- `--output_path`: Where to save the master flat FITS file
- `--bias_level`: Bias level to subtract (ADU)
- `--crop_size`: Size for edge cropping (pixels)

### 2. Create Position Stacks

Stack multiple exposures for each telescope pointing position:

```bash
pesto-create-stack \
    --filter Ha \
    --name Arp94 \
    --output_dir /path/to/output \
    --data_root /path/to/raw/data \
    --n_pos 6
```

**Parameters:**
- `--filter`: Filter name (e.g., Ha, i, r, g)
- `--name`: Object/field name
- `--output_dir`: Directory for output stacks
- `--data_root`: Root directory containing raw observation data
- `--n_pos`: Number of pointing positions

### 3. Generate Mosaic

Combine all position stacks into a final mosaic image:

```bash
pesto-mosaic \
    --filter i \
    --name Arp94 \
    --output_dir /path/to/output \
    --n_pos 6 \
    --combine median
```

**Parameters:**
- `--filter`: Filter name matching the stacks
- `--name`: Object/field name matching the stacks
- `--output_dir`: Directory containing stacks and for final mosaic
- `--n_pos`: Number of positions to combine
- `--combine`: Combination method (median, mean, etc.)

## Setup

### Astrometry Index Files

The pipeline requires Astrometry.net index files for plate solving. Download the index-5200 series:

```bash
#!/bin/bash

BASE_URL="http://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE"

for i in $(seq -w 0 47); do
    FILE="index-5200-${i}.fits"
    echo "Downloading $FILE ..."
    wget -c "${BASE_URL}/${FILE}"
done
```

Place the downloaded index files in your Astrometry.net index directory (typically `~/.local/share/astrometry` or `/usr/local/astrometry/data`).

## Example Workflow

Complete reduction for observations from 2024-11-09:

```bash
# Step 1: Create master flat
pesto-create-flat \
    --data_dir /home/carterrhea/Documents/OMM-Data/241109/DomeFlat \
    --output_path /home/carterrhea/Documents/OMM-Clean/241109/master_flat.fits \
    --bias_level 300

# Step 2: Create stacks for each position
pesto-create-stack \
    --filter Ha \
    --name Arp94 \
    --output_dir /home/carterrhea/Documents/OMM-Clean/241109/Arp94 \
    --data_root /home/carterrhea/Documents/OMM-Data/241109 \
    --n_pos 6

# Step 3: Generate final mosaic
pesto-mosaic \
    --filter Ha \
    --name Arp94 \
    --output_dir /home/carterrhea/Documents/OMM-Clean/241109/Arp94 \
    --n_pos 6 \
    --combine median
```

## Development

### Running Tests

Install test dependencies:

```bash
pip install -e ".[test]"
```

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=pesto_reduction --cov-report=html
```

For more details on testing, see [tests/README.md](tests/README.md).

## Project Structure

```
pesto-reduction/
├── src/
│   └── pesto_reduction/
│       ├── cli/              # Command-line interface modules
│       ├── create_stack.py   # Stacking functionality
│       ├── flats.py          # Flat-field processing
│       ├── mosaic.py         # Mosaic generation
│       ├── utils.py          # Utility functions
│       └── logger.py         # Logging configuration
├── tests/                    # Unit tests
├── pyproject.toml            # Project configuration
├── pytest.ini                # Pytest configuration
└── README.md                 # This file
```

## Author

Carter Rhea (carterrhea93@gmail.com)

## License

See LICENSE file for details.
