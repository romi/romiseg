#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Convert PyTorch models to state dictionary format.

This script can be used to convert PyTorch models to state dictionary format, which is a simpler format
that can be used to share models between different platforms.

The script supports both single model files and directories containing multiple model files.

## Usage

```shell
python convert_models.py <input> [--output <output>] [--pattern <pattern>] [--recursive] [--overwrite] [--log-level <level>]
python convert_models.py --help
python convert_models.py --version
```

## Example usage

```python
>>> from romiseg.cli.convert_models import setup_module_alias
>>> from romiseg.cli.convert_models import convert_model_to_state_dict
>>> from plantdb.commons import io
>>> from plantdb.commons.test_database import test_database
>>> from romiseg.models.unet import ResNetUNet
>>> # Set up a test database with an old API model with weights_only=True
>>> db = test_database('real_plant', with_models=True)
>>> db.connect()
>>> # Get the model fileset and model file
>>> model_name = 'Resnet_896_896_epoch50.pt'
>>> models_fileset = db.get_scan('models').get_fileset('models')
>>> models_file = models_fileset.get_file(model_name.split('.')[0])
>>> # Get the label names from the model metadata
>>> label_names = models_file.get_metadata('label_names')
>>> print(label_names)
>>> # Set up the module alias and convert the model to state dict
>>> setup_module_alias()
>>> state_model_file = convert_model_to_state_dict(models_file.path())
>>> # Check the state dict was saved correctly
>>> assert state_model_file.exists()
>>> # Load the state dict and instantiate model
>>> model = ResNetUNet(len(label_names))
>>> model.load_state_dict(io.read_torch(state_model_file))
>>> # Clean up
>>> db_path = db.path()
>>> db.disconnect()
>>> import shutil
>>> shutil.rmtree(db_path)
```
"""

import argparse
import logging
import sys
import types
from pathlib import Path

import torch
from colorlog import ColoredFormatter

# Configure logger
logger = logging.getLogger('convert_models')

# Define the log message format for colored logs.
# The color is dynamically applied using `log_color` and `bg_blue` and reset after styling.
COLOR_LOG_FMT = "{log_color}{levelname:<8}{reset} {bg_blue}[{name}]{reset} {message}"

# Create a colored logging formatter instance for enhanced log readability in terminal outputs.
# Applies colors for log levels, resets the style after application, and uses the same `{}` style formatting.
COLORED_FORMATTER = ColoredFormatter(
    COLOR_LOG_FMT,
    datefmt=None,  # No date is included in the log format.
    reset=True,  # Automatically reset styles applied to the log after each log message.
    style='{',  # Use the `{}` style of string formatting.
)


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    # Get the root logger
    logger.setLevel(level)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a console handler and set its formatter to the colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(COLORED_FORMATTER)

    # Add the handler to the root logger
    logger.addHandler(console_handler)


def setup_module_alias():
    """Create temporary module alias for backward compatibility.
    Old API was using a class named 'ResNetUNet' in the 'utils.segmentation_model' module.
    New API uses a class named 'ResNetUNet' in the 'models.unet' module.
    """
    # Create a dummy module
    dummy_module = types.ModuleType('romiseg.utils.segmentation_model')
    # Import the actual class from its new location
    from romiseg.models.unet import ResNetUNet
    # Add the class to the dummy module
    dummy_module.ResNetUNet = ResNetUNet
    # Add the module to sys.modules
    sys.modules['romiseg.utils.segmentation_model'] = dummy_module
    logger.info("Module alias created for backward compatibility")


def extract_model_state_dict(loaded_obj):
    """Extract state dictionary from a loaded model object.
    Parameters
    ----------
    loaded_obj : object
        The loaded object from torch.load
    Returns
    -------
    dict or None
        The state dictionary if successful, None otherwise
    """
    # Determine if the loaded object is a tuple containing the model
    if isinstance(loaded_obj, tuple) and hasattr(loaded_obj[0], 'state_dict'):
        model = loaded_obj[0]
        logger.info("Extracted model from tuple")
    # Or if it's the model directly
    elif hasattr(loaded_obj, 'state_dict'):
        model = loaded_obj
        logger.info("Loaded object is the model")
    # Or if it's a dict with a 'model' key
    elif isinstance(loaded_obj, dict) and 'model' in loaded_obj and hasattr(loaded_obj['model'], 'state_dict'):
        model = loaded_obj['model']
        logger.info("Extracted model from dictionary")
    else:
        logger.error(f"Could not identify model in loaded object of type: {type(loaded_obj)}")
        return None

    # Extract state dict
    state_dict = model.state_dict()
    logger.info(f"Extracted state_dict with {len(state_dict)} entries")
    return state_dict


def save_state_dict(state_dict, output_path):
    """Save a state dictionary to disk.
    Parameters
    ----------
    state_dict : dict
        The state dictionary to save
    output_path : Path
        Path where to save the state dictionary
    Returns
    -------
    Path
        The path where the state dictionary was saved
    """
    torch.save(state_dict, output_path)
    logger.info(f"Saved state_dict to: {output_path}")
    return output_path


def convert_model_to_state_dict(input_path, output_path=None, overwrite=False):
    """Load a full model file and save just its state dictionary
    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the saved model file
    output_path : str, optional
        Path to save the state dict. If None, will append '_state_dict' to the input path
    overwrite : bool, optional
        Whether to overwrite existing files
    Returns
    -------
    str
        Path to the saved state dict file
    """
    if not isinstance(input_path, Path):
        input_path = Path(input_path)
    # Generate output path if not provided
    if output_path is None:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_state_dict{input_path.suffix}"
    else:
        output_path = Path(output_path)

    # Check if output file already exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file {output_path} already exists. Use --overwrite to force.")

    logger.info(f"Loading model from: {input_path}")
    try:
        # Try loading with weights_only=False (less secure but needed in this case)
        loaded_obj = torch.load(input_path, weights_only=False)
        logger.info("Model loaded successfully using `weights_only=False`")
    except Exception as e:
        logger.error(f"Error loading model with `weights_only=False`: {e}")
        return None

    # Extract the state dictionary
    state_dict = extract_model_state_dict(loaded_obj)
    if state_dict is None:
        return None

    # Save the state dictionary
    return save_state_dict(state_dict, output_path)


def process_directory(directory, output_dir=None, pattern="*.pt", recursive=False, overwrite=False):
    """Process all model files in a directory
    Parameters
    ----------
    directory : str
        Directory containing model files
    output_dir : str, optional
        Directory to save state dicts. If None, saves in the same directory
    pattern : str
        File pattern to match (default: "*.pt")
    recursive : bool
        Whether to search subdirectories
    overwrite : bool
        Whether to overwrite existing files
    """
    directory = Path(directory)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    # Find all model files
    if recursive:
        model_files = list(directory.glob(f"**/{pattern}"))
    else:
        model_files = list(directory.glob(pattern))

    logger.info(f"Found {len(model_files)} model files")
    converted = 0
    failed = 0
    skipped = 0
    for model_file in model_files:
        # Skip files that look like they're already state dicts
        if "_state_dict" in model_file.stem:
            logger.info(f"Skipping {model_file} (appears to be a state dict already)")
            skipped += 1
            continue
        # Determine output path
        if output_dir:
            rel_path = model_file.relative_to(directory)
            output_path = output_dir / rel_path.parent / f"{rel_path.stem}_state_dict{rel_path.suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None  # Will be auto-generated in convert_model_to_state_dict
        try:
            result = convert_model_to_state_dict(model_file, output_path, overwrite)
            if result:
                converted += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error converting {model_file}: {e}")
            failed += 1

    logger.info(f"Conversion summary:")
    logger.info(f"- Converted: {converted}")
    logger.info(f"- Failed: {failed}")
    logger.info(f"- Skipped: {skipped}")


def parser():
    """Parses command-line arguments.

    Returns
    -------
    argparse.ArgumentParser
        A configured argument parser with an option for specifying the configuration directory.
    """
    parser = argparse.ArgumentParser(description="Convert PyTorch models to state dictionary format")
    parser.add_argument("input", help="Input model file or directory")
    parser.add_argument("--output", help="Output file or directory (optional)")
    parser.add_argument("--pattern", default="*.pt",
                        help="File pattern to match when input is a directory (default: *.pt)")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")

    return parser


def main():
    args = parser().parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)

    # Set up module alias for backward compatibility
    setup_module_alias()

    input_path = Path(args.input)

    if input_path.is_file():
        # Process a single file
        convert_model_to_state_dict(input_path, args.output, args.overwrite)
    elif input_path.is_dir():
        # Process a directory
        process_directory(input_path, args.output, args.pattern, args.recursive, args.overwrite)
    else:
        logger.error(f"Input path {input_path} does not exist")


if __name__ == "__main__":
    main()
