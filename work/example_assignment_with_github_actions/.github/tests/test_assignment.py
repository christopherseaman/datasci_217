# test assignment7
import os
import pytest
import json


# Helper to parse metrics file
def parse_metrics(filepath):
    metrics = {}
    with open(filepath) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                # Try to parse as float or list
                if v.startswith("["):
                    import ast

                    metrics[k] = ast.literal_eval(v)
                else:
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        metrics[k] = v
    return metrics


def test_part1_files_exist():
    assert os.path.exists("models/best_model.keras") or os.path.exists("models/best_model.pt")
    assert os.path.exists("results/part_1/model_comparison.txt")


def test_part1_metrics_format():
    metrics = parse_metrics("results/part_1/model_comparison.txt")
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]:
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    
    # Check if cross-validation results are present
    assert "model_1_mean_accuracy" in metrics
    assert "model_2_mean_accuracy" in metrics
    assert "model_3_mean_accuracy" in metrics


def test_part2_files_exist():
    assert os.path.exists("utils/llm_chat.py")
    assert os.path.exists("results/part_2/usage_examples.txt")


def test_part2_utility_functionality():
    # Check if the utility file has required functions/classes
    with open("utils/llm_chat.py", "r") as f:
        content = f.read()
        assert "class LLMClient" in content
        assert "class LLMChatTool" in content
        assert "def main" in content
    
    # Check if the utility is executable
    assert os.access("utils/llm_chat.py", os.X_OK)


def test_part3_files_exist():
    assert os.path.exists("utils/structured_response.py")
    assert os.path.exists("results/part_3/prompt_comparison.txt")


def test_part3_metrics_format():
    metrics = parse_metrics("results/part_3/prompt_comparison.txt")
    
    # Check if at least one prompting strategy is evaluated
    strategy_found = False
    for strategy in ["zero-shot", "one-shot", "few-shot"]:
        if f"{strategy}_accuracy" in metrics or f"{strategy.capitalize()} Prompting" in metrics:
            strategy_found = True
            break
    
    assert strategy_found, "No prompting strategy evaluation found"


def test_part4_files_exist():
    """Optional test for part 4 - only runs if files exist"""
    if not os.path.exists("models/nanogpt.pt"):
        pytest.skip("Part 4 (optional) not implemented")
    
    assert os.path.exists("results/part_4/training_metrics.txt")
    assert os.path.exists("results/part_4/generation_evaluation.txt")


def test_part4_metrics_format():
    """Optional test for part 4 - only runs if files exist"""
    if not os.path.exists("results/part_4/training_metrics.txt"):
        pytest.skip("Part 4 (optional) not implemented")
    
    metrics = parse_metrics("results/part_4/training_metrics.txt")
    
    # Check if training metrics are present
    assert "Final Train Loss" in metrics or "Final_Train_Loss" in metrics
    assert "Final Validation Loss" in metrics or "Final_Validation_Loss" in metrics
    
    # Check if generation evaluation metrics are present
    with open("results/part_4/generation_evaluation.txt", "r") as f:
        content = f.read()
        assert "Generated Samples" in content
        assert "Evaluation Metrics" in content