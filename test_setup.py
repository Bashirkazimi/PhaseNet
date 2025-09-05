"""Simple test to verify PhaseNet setup."""

import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import torchvision
        print(f"✅ Torchvision {torchvision.__version__}")
    except ImportError:
        print("❌ Torchvision not found")
        return False
    
    try:
        import lightning
        print(f"✅ Lightning {lightning.__version__}")
    except ImportError:
        print("❌ Lightning not found")
        return False
    
    try:
        import hydra
        print(f"✅ Hydra available")
    except ImportError:
        print("❌ Hydra not found")
        return False
    
    return True


def test_phase_modules():
    """Test that PhaseNet modules can be imported."""
    print("\nTesting PhaseNet modules...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from src.data.phase_stem_datamodule import PhaseSTEMDataModule
        print("✅ PhaseSTEMDataModule")
    except ImportError as e:
        print(f"❌ PhaseSTEMDataModule: {e}")
        return False
    
    try:
        from src.models.phase_classification_module import PhaseClassificationModule
        print("✅ PhaseClassificationModule")
    except ImportError as e:
        print(f"❌ PhaseClassificationModule: {e}")
        return False
    
    try:
        from src.models.components.phase_networks import create_phase_network
        print("✅ Phase networks")
    except ImportError as e:
        print(f"❌ Phase networks: {e}")
        return False
    
    return True


def test_configs():
    """Test that configuration files exist."""
    print("\nTesting configuration files...")
    
    configs_dir = Path(__file__).parent / "configs"
    
    required_configs = [
        "train_phase_classification.yaml",
        "data/phase_stem.yaml",
        "model/phase_classification.yaml",
        "experiment/phase_classification_example.yaml"
    ]
    
    all_exist = True
    for config in required_configs:
        config_path = configs_dir / config
        if config_path.exists():
            print(f"✅ {config}")
        else:
            print(f"❌ {config} not found")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("🧪 PhaseNet Setup Test\n")
    
    tests = [
        ("Basic imports", test_imports),
        ("PhaseNet modules", test_phase_modules),
        ("Configuration files", test_configs),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        all_passed = all_passed and result
    
    if all_passed:
        print("\n🎉 All tests passed! PhaseNet is ready to use.")
        print("\nTo get started with phase classification:")
        print("1. Prepare your 4D-STEM data in the required format")
        print("2. Run: python src/train.py -cn train_phase_classification")
    else:
        print("\n⚠️  Some tests failed. Please check the installation.")


if __name__ == "__main__":
    main()
