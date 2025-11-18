#!/usr/bin/env python3
"""
Test script to verify the ID-based annotation system works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.annotations_loader import AnnotationConfigLoader
from core.few_shot_manager import MessageManager

def test_annotation_loader():
    """Test AnnotationConfigLoader with ID-based structure"""
    print("=" * 80)
    print("Testing AnnotationConfigLoader")
    print("=" * 80)

    # Load annotations
    loader = AnnotationConfigLoader("data/annotations.json")

    # Test basic loading
    examples = loader.get_all_examples()
    print(f"\n✅ Loaded {len(examples)} examples")

    # Test ID-based loading
    example_1 = loader.get_example_by_id(1)
    if example_1:
        print(f"\n✅ get_example_by_id(1) works:")
        print(f"   ID: {example_1['id']}")
        print(f"   Expected response: {example_1['expected_response'][:50]}...")
    else:
        print(f"\n❌ Failed to load example by ID")
        return False

    # Test getting examples by IDs
    examples_by_ids = loader.get_examples_by_ids([1, 2, 3])
    print(f"\n✅ get_examples_by_ids([1, 2, 3]) returned {len(examples_by_ids)} examples")

    # Test getting example IDs for task
    example_ids = loader.get_example_ids_for_task(
        "serve_detection",
        num_examples=5
    )
    print(f"\n✅ get_example_ids_for_task():")
    print(f"   Example IDs: {example_ids}")

    # Test validation
    validation_errors = loader.validate_all_examples()
    if validation_errors:
        print(f"\n❌ Validation errors found:")
        for identifier, errors in validation_errors.items():
            print(f"   {identifier}: {errors}")
        return False
    else:
        print(f"\n✅ All examples passed validation")

    return True


def test_message_manager():
    """Test MessageManager with ID-based loading"""
    print("\n" + "=" * 80)
    print("Testing MessageManager")
    print("=" * 80)

    manager = MessageManager("few_shot_examples")

    # Test loading example by ID
    example_1 = manager.load_example_by_id(1)
    if example_1:
        frames = example_1['frames']
        metadata = example_1['metadata']
        print(f"\n✅ load_example_by_id(1) works:")
        print(f"   Frames: {len(frames)}")
        print(f"   ID: {metadata.get('id')}")
        print(f"   Expected response: {metadata.get('expected_response', 'N/A')[:50]}...")
    else:
        print(f"\n❌ Failed to load example by ID")
        return False

    # Test loading multiple examples by IDs
    examples = manager.load_examples_by_ids([1, 2, 3])
    print(f"\n✅ load_examples_by_ids([1, 2, 3]) returned {len(examples)} examples")
    total_frames = sum(len(ex['frames']) for ex in examples)
    print(f"   Total frames: {total_frames}")

    # Test that all example metadata has ID and expected_response
    all_have_id = all('id' in ex['metadata'] for ex in examples)
    all_have_response = all('expected_response' in ex['metadata'] for ex in examples)

    if all_have_id and all_have_response:
        print(f"\n✅ All examples have 'id' and 'expected_response' in metadata")
    else:
        print(f"\n❌ Some examples missing 'id' or 'expected_response'")
        return False

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ID-BASED ANNOTATION SYSTEM TEST")
    print("=" * 80)

    success = True

    # Test 1: AnnotationConfigLoader
    if not test_annotation_loader():
        print("\n❌ AnnotationConfigLoader tests failed")
        success = False

    # Test 2: MessageManager
    if not test_message_manager():
        print("\n❌ MessageManager tests failed")
        success = False

    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe ID-based annotation system is working correctly!")
        print("You can now use:")
        print("  - data/annotations.json for configuration")
        print("  - few_shot_examples/{id}/ for example storage")
        print("=" * 80)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
