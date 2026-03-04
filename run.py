#!/usr/bin/env python3
"""Quick start script for Activation Maximization XAI project."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print("Error:", e.stderr)
        return False

def main():
    """Main function to run the project."""
    print("🧠 Activation Maximization XAI Project - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   The 'src' directory should be present.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10+ is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print("✅ Python version check passed")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -e .", "Installing project dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ["data", "checkpoints", "logs", "assets/visualizations"]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_command("python -m pytest tests/ -v", "Running unit tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Train model
    print("\n🚀 Training model...")
    if not run_command("python scripts/train.py --config configs/config.yaml", "Training model"):
        print("❌ Failed to train model")
        sys.exit(1)
    
    # Run activation maximization
    print("\n🔍 Running activation maximization...")
    if not run_command("python scripts/train.py --config configs/config.yaml --skip_training", "Running activation maximization"):
        print("❌ Failed to run activation maximization")
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Launch the interactive demo:")
    print("   streamlit run demo/streamlit_app.py")
    print("\n2. Or run the example notebook:")
    print("   jupyter notebook notebooks/activation_maximization_example.ipynb")
    print("\n3. Check the generated visualizations in:")
    print("   assets/visualizations/")
    
    print("\n⚠️  Remember: This is for research and educational use only!")
    print("   Always validate results with domain experts.")

if __name__ == "__main__":
    main()
