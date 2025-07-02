#!/usr/bin/env python3
"""
CLI entry point for the Concept Summary Agent.
This script allows running the ConceptSummaryAgent from the command line
for testing and manual execution.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the shared directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "shared"))

try:
    from aclarai_shared.concept_summary_agent import ConceptSummaryAgent
    from aclarai_shared.config import load_config
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run the Concept Summary Agent to generate concept pages"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (defaults to settings/aclarai.config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually generating files",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        if args.config:
            config = load_config(config_file=args.config)
        else:
            config = load_config()
        
        logger.info(f"Vault path: {config.paths.vault}")
        logger.info(f"Concepts directory: {config.paths.tier3}")
        logger.info(f"Model: {getattr(config.concept_summaries, 'model', 'default')}")
        logger.info(f"Max examples: {getattr(config.concept_summaries, 'max_examples', 5)}")
        logger.info(f"Skip if no claims: {getattr(config.concept_summaries, 'skip_if_no_claims', True)}")
        
        if args.dry_run:
            logger.info("DRY RUN MODE: Would create ConceptSummaryAgent and run it")
            logger.info("No files will be generated in dry run mode")
            return 0
        
        # Create and run the agent
        logger.info("Creating ConceptSummaryAgent...")
        agent = ConceptSummaryAgent(config=config)
        
        logger.info("Running ConceptSummaryAgent...")
        result = agent.run_agent()
        
        # Print results
        print("\n" + "="*50)
        print("CONCEPT SUMMARY AGENT RESULTS")
        print("="*50)
        print(f"Success: {result['success']}")
        print(f"Concepts processed: {result['concepts_processed']}")
        print(f"Concepts generated: {result['concepts_generated']}")
        print(f"Concepts skipped: {result['concepts_skipped']}")
        print(f"Errors: {result['errors']}")
        
        if result['error_details']:
            print("\nError details:")
            for error in result['error_details']:
                print(f"  - {error}")
        
        if result['success']:
            logger.info("ConceptSummaryAgent completed successfully")
            return 0
        else:
            logger.error("ConceptSummaryAgent failed")
            return 1
            
    except Exception as e:
        logger.error(f"Failed to run ConceptSummaryAgent: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())