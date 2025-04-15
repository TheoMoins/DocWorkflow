#!/usr/bin/env python
"""
Main entry point for the document analysis tool.
Provides access to layout segmentation and line segmentation.
"""
import sys
import argparse
from layout.cli import LayoutCLI
from line.cli import LineCLI

def main():
    """Main entry point for the document analysis tool."""
    parser = argparse.ArgumentParser(
        description="Document analysis tool for layout and line segmentation"
    )
    parser.add_argument('module', choices=['layout', 'line'],
                       help="Module to run: 'layout' for layout segmentation, "
                            "'line' for line segmentation")
    
    # Parse only the first argument to determine which module to use
    args, remaining = parser.parse_known_args()
    
    if args.module == 'layout':
        # Run the layout segmentation CLI interface
        cli = LayoutCLI()
        cli.run(remaining)
    elif args.module == 'line':
        # Run the line segmentation CLI interface
        cli = LineCLI()
        cli.run(remaining)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()