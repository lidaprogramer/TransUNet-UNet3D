"""
Standalone script for comprehensive post-processing analysis
Usage: python run_postprocessing_analysis.py [--volume-filter PATTERN]
"""

import argparse
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def main():
    parser = argparse.ArgumentParser(description="Comprehensive coronary artery post-processing analysis")
    parser.add_argument("--volume-filter", type=str, default=None,
                       help="Filter volumes by name prefix (e.g. 'extracted_601')")
    parser.add_argument("--output-dir", type=str, default="./postprocessing_analysis",
                       help="Output directory for results")
    parser.add_argument("--save-all", action="store_true",
                       help="Save all volumes, not just best/worst")
    parser.add_argument("--min-component-size", type=int, default=50,
                       help="Minimum component size")
    parser.add_argument("--closing-radius", type=int, default=2,
                       help="Morphological closing radius")
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Post-Processing Analysis")
    print("="*60)
    
    # Run the enhanced test with post-processing
    # This would call the enhanced test_imagecas_with_postprocessing logic
    # with the specified parameters
    
    print(f"Volume filter: {args.volume_filter or 'All volumes'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Component size threshold: {args.min_component_size}")
    print(f"Closing radius: {args.closing_radius}")
    
    # Import and run the analysis
    try:
        from test_imagecas_with_postprocessing import run_analysis
        run_analysis(args)
    except ImportError:
        print("⚠️  Please ensure test_imagecas_with_postprocessing.py is available")
        print("    Run: python test_imagecas_with_postprocessing.py")

if __name__ == "__main__":
    main()
