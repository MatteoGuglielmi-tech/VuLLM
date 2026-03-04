import sys

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_torch():
    """Test PyTorch installation."""
    print_section("Testing PyTorch")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test transformers installation."""
    print_section("Testing Transformers")
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        return False

def test_unsloth():
    """Test unsloth installation and load a model."""
    print_section("Testing Unsloth")
    try:
        from unsloth import FastLanguageModel
        print("✓ Unsloth imported successfully")

        # Load a small Qwen model
        print("\nLoading Qwen2.5-Coder-0.5B model with unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            #model_name="unsloth/Qwen2.5-Coder-0.5B-Instruct",
            model_name= "unsloth/llama-3.1-8b-instruct-bnb-4bit",
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization for efficiency
        )
        print("✓ Model loaded successfully")

        model_type = getattr(model.config, "model_type", "").lower()
        print(model.config)
        print(model_type)

        return True
    except Exception as e:
        print(f"✗ Unsloth test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  Installation Test Script")
    print("  Testing: torch, transformers, unsloth")
    print("="*60)

    results = {
        'unsloth': test_unsloth(),
        'torch': test_torch(),
        'transformers': test_transformers(),
    }

    # Summary
    print_section("Test Summary")
    all_passed = True
    for package, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{package:15} : {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("  ✓ All tests passed!")
    else:
        print("  ✗ Some tests failed. Check output above.")
    print("="*60 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
