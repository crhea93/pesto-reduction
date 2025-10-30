# Testing Guide for pesto-reduction

## Overview

The pesto-reduction project includes a comprehensive test suite with **161 tests** achieving **86% code coverage**.

## Test Statistics

| Metric | Value |
|--------|-------|
| **Total Tests** | 161 |
| **Pass Rate** | 100% ✅ |
| **Overall Coverage** | 86% |
| **Time to Run** | ~6 seconds |

## Module Coverage

| Module | Coverage | Lines |
|--------|----------|-------|
| **logger.py** | 100% | 20/20 |
| **utils.py** | 99% | 82/83 |
| **cli/create_flat.py** | 93% | 14/15 |
| **cli/create_stack.py** | 96% | 27/28 |
| **cli/mosaic.py** | 96% | 27/28 |
| **flats.py** | 93% | 42/45 |
| **mosaic.py** | 85% | 199/234 |
| **create_stack.py** | 57% | 39/68 |

## Test Files

### 1. test_utils.py (68 tests)
Tests for core utility functions including:
- Coordinate resolution with `get_coords()`
- Cosmic ray removal with `remove_cosmic()`
- Background subtraction with `subtract_background()`
- NaN interpolation with `fill_nan_interpolation()`
- Robust 2D background estimation with `robust_background_2d()`
- Astrometry field solving with `solve_field()` (8 tests)

**Coverage**: 99%

### 2. test_flats.py (11 tests)
Tests for flat-field processing:
- Loading flat frames with bias subtraction
- Histogram visualization
- Cosmic ray removal for flats
- Master flat creation with normalization and outlier rejection

**Coverage**: 93%

### 3. test_cli_create_flat.py (14 tests)
Tests for the create-flat CLI:
- Argument parsing with various parameter combinations
- Main function behavior and parameter passing
- Default values and error handling

**Coverage**: 93%

### 4. test_cli_create_stack.py (15 tests)
Tests for the create-stack CLI:
- Argument parsing with filter/name/directory combinations
- Position naming and directory iteration
- Flat file verification
- Multiple position handling

**Coverage**: 96%

### 5. test_cli_mosaic.py (19 tests)
Tests for the mosaic CLI:
- Argument parsing for combine methods
- Image path construction
- Output path generation
- Combine method selection (median/mean)

**Coverage**: 96%

### 6. test_create_stack.py (9 tests)
Tests for image stacking core functionality:
- WCS grid creation with various parameters
- Light frame processing with mocking
- Error handling for missing files

**Coverage**: 57%

### 7. test_create_stack_module.py (15 tests) ✨ NEW
Comprehensive tests for create_stack module:
- WCS creation at different declinations
- Pixel scale variations
- Field size handling
- Edge cases and coordinate systems

**Coverage**: Complements test_create_stack.py

### 8. test_mosaic.py (24 tests)
Core mosaic functionality tests:
- Background 2D estimation
- Smooth weight map creation
- Photometric matching of images
- Seamless image combination
- Footprint visualization

**Coverage**: 85%

### 9. test_mosaic_advanced.py (31 tests) ✨ NEW
Advanced mosaic tests including:
- Gradient background handling
- Variable noise levels
- Multiple bright sources
- Circular and striped NaN patterns
- Photometric matching with noise and offsets
- Multi-image combinations (3-4 images)
- Quality assessment functionality
- Large and small image processing

**Coverage**: Extends mosaic coverage to 85%

## Running Tests

### Install Dependencies
```bash
pip install -e ".[test]"
```

### Run All Tests
```bash
pytest
```

### Run with Coverage Report
```bash
pytest --cov=pesto_reduction --cov-report=term-missing
```

### Run Specific Test File
```bash
pytest tests/test_utils.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_utils.py::TestGetCoords -v
```

### Run Specific Test
```bash
pytest tests/test_utils.py::TestGetCoords::test_get_coords_success -v
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Print Statements
```bash
pytest -s
```

## Test Organization

Tests are organized into logical groups by module and functionality:

```
tests/
├── test_utils.py              # Utility functions (68 tests)
├── test_flats.py              # Flat-field processing (11 tests)
├── test_cli_create_flat.py    # CLI for creating flats (14 tests)
├── test_cli_create_stack.py   # CLI for stacking (15 tests)
├── test_cli_mosaic.py         # CLI for mosaicing (19 tests)
├── test_create_stack.py       # Image stacking core (9 tests)
├── test_create_stack_module.py# Stacking module advanced (15 tests)
├── test_mosaic.py             # Mosaic core (24 tests)
├── test_mosaic_advanced.py    # Mosaic advanced (31 tests)
├── conftest.py                # Shared fixtures
└── README.md                  # Testing documentation
```

## Test Coverage Areas

### ✅ Comprehensive Coverage

1. **Normal Operation**: Standard input, typical parameters
2. **Edge Cases**: Empty arrays, NaN values, single elements, extreme values
3. **Error Handling**: Missing files, invalid inputs, exceptions
4. **Parameter Variations**: Different combinations, boundary values
5. **Data Shapes**: Various image sizes, from 16x16 to 1000x1000

### Mocking Strategy

- External dependencies mocked (subprocess, file I/O)
- Expensive operations (astronomy library calls) mocked
- No network calls in tests
- Fixture-based test data

### Key Testing Features

✨ **Fixtures** (conftest.py)
- Sample images (basic, with sources, with NaN values)
- FITS files with headers
- Cosmic ray images
- Weight maps

✨ **Parametrized Tests**
- Multiple parameter combinations tested efficiently
- Different declinations, pixel scales, field sizes
- Various image shapes and noise levels

✨ **Integration Tests**
- Multi-step workflows tested
- Background estimation + weight maps
- Image combination with corrections

## Continuous Integration

Tests are designed for CI/CD pipelines:
- Fast execution (~6 seconds)
- No external dependencies
- Deterministic results
- Clear pass/fail status

## Future Coverage Improvements

Areas for potential enhancement:

1. **create_stack.py** (57% → 80%+)
   - Parallel processing tests
   - Filter matching logic
   - Full integration workflows

2. **mosaic.py** (85% → 95%+)
   - Quality assessment visualization
   - Detailed reprojection tests
   - WCS handling edge cases

## Contributing Tests

When adding new features:

1. Write tests alongside code
2. Aim for >90% coverage
3. Use descriptive test names
4. Include docstrings explaining test purpose
5. Test both happy path and error cases
6. Use fixtures for common test data
7. Mock external dependencies

## Example Test Structure

```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_operation(self):
        """Test basic case."""
        # Arrange
        input_data = ...
        
        # Act
        result = new_feature(input_data)
        
        # Assert
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case."""
        # Test with empty, NaN, or extreme values
        
    @patch('module.external_function')
    def test_with_mock(self, mock_function):
        """Test with mocked dependency."""
        mock_function.return_value = ...
```

## Debugging Failed Tests

```bash
# Run failing test with full output
pytest tests/test_file.py::TestClass::test_name -vv -s

# Show local variables on failure
pytest --showlocals tests/test_file.py

# Drop into debugger on failure
pytest --pdb tests/test_file.py

# Run only failed tests from last run
pytest --lf
```

## Test Metrics Summary

| Metric | Value |
|--------|-------|
| Total Lines of Test Code | ~4,000+ |
| Test Classes | 40+ |
| Test Functions | 161 |
| Average Test Duration | ~37ms |
| Slowest Test | ~200ms |
| Fastest Test | ~1ms |

---

**Last Updated**: October 2025
**Test Framework**: pytest
**Coverage Tool**: pytest-cov
