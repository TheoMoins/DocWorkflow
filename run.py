#!/usr/bin/env python

import sys
import argparse
from core.cli import CLI

def main():
    """
    Main entry point for the document analysis tool.
    """
    parser = argparse.ArgumentParser(
        description="Document analysis tool for layout and line segmentation"
    )
    parser.add_argument('module', choices=['layout', 'line'],
                       help="Module to run: 'layout' for layout segmentation, "
                            "'line' for line segmentation")
    
    # Parse only the first argument to determine which module to use
    args, remaining = parser.parse_known_args()
    
    if args.module in ['layout', 'line']:
        cli = CLI(args.module)
        cli.run(remaining)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()