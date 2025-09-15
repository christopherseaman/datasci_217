# Part 3: Prompt Engineering Basics

## Introduction

In this part, you'll experiment with different prompting techniques to improve the quality of responses from Large Language Models (LLMs). You'll compare zero-shot, one-shot, and few-shot prompting approaches and document which works best for different types of questions.

## Learning Objectives

- Understand different prompting techniques
- Compare zero-shot, one-shot, and few-shot prompting
- Analyze the impact of prompt design on response quality

## Setup

```python
# Import necessary libraries
import requests
import json
```

## 1. Understanding Prompting Techniques

LLMs can be prompted in different ways to get better responses:

1. **Zero-shot prompting**: Asking the model a question directly without examples
2. **One-shot prompting**: Providing one example before asking your question
3. **Few-shot prompting**: Providing multiple examples before asking your question

## 2. Creating Prompting Templates

Your first task is to create templates for different prompting strategies.

```python
# Define a question to experiment with
question = "What foods should be avoided by patients with gout?"

# Example for one-shot and few-shot prompting
example_q = "What are the symptoms of gout?"
example_a = "Gout symptoms include sudden severe pain, swelling, redness, and tenderness in joints, often the big toe."

# Examples for few-shot prompting
examples = [
    ("What are the symptoms of gout?",
     "Gout symptoms include sudden severe pain, swelling, redness, and tenderness in joints, often the big toe."),
    ("How is gout diagnosed?",
     "Gout is diagnosed through physical examination, medical history, blood tests for uric acid levels, and joint fluid analysis to look for urate crystals.")
]

# TODO: Create prompting templates
# Zero-shot template (just the question)
zero_shot_template = "Question: {question}\nAnswer:"

# One-shot template (one example + the question)
one_shot_template = """Question: {example_q}
Answer: {example_a}

Question: {question}
Answer:"""

# Few-shot template (multiple examples + the question)
few_shot_template = """Question: {examples[0][0]}
Answer: {examples[0][1]}

Question: {examples[1][0]}
Answer: {examples[1][1]}

Question: {question}
Answer:"""

# TODO: Format the templates with your question and examples
zero_shot_prompt = zero_shot_template.format(question=question)
one_shot_prompt = one_shot_template.format(example_q=example_q, example_a=example_a, question=question)
# For few-shot, you'll need to format it with the examples list
few_shot_prompt = few_shot_template.format(examples=examples, question=question)

print("Zero-shot prompt:")
print(zero_shot_prompt)
print("\nOne-shot prompt:")
print(one_shot_prompt)
print("\nFew-shot prompt:")
print(few_shot_prompt)
```

## 3. Connecting to the LLM API

Next, implement a function to send prompts to an LLM API and get responses.

```python
def get_llm_response(prompt, model_name="google/flan-t5-base", api_key=None):
    """Get a response from the LLM based on the prompt"""
    # TODO: Implement the get_llm_response function
    pass

# TODO: Test your get_llm_response function with different prompts
```

## 4. Comparing Prompting Strategies

Now, let's compare the different prompting strategies on a set of healthcare questions.

```python
# List of healthcare questions to test
questions = [
    "What foods should be avoided by patients with gout?",
    "What medications are commonly prescribed for gout?",
    "How can gout flares be prevented?",
    "Is gout related to diet?",
    "Can gout be cured permanently?"
]

# TODO: Compare the different prompting strategies on these questions
# For each question:
# - Create prompts using each strategy
# - Get responses from the LLM
# - Store the results
```

## 5. Evaluating Responses

Create a simple evaluation function to score the responses based on the presence of expected keywords.

```python
def score_response(response, keywords):
    """Score a response based on the presence of expected keywords"""
    # TODO: Implement the score_response function
    # Example implementation:
    response = response.lower()
    found_keywords = 0
    for keyword in keywords:
        if keyword.lower() in response:
            found_keywords += 1
    return found_keywords / len(keywords) if keywords else 0

# Expected keywords for each question
expected_keywords = {
    "What foods should be avoided by patients with gout?": 
        ["purine", "red meat", "seafood", "alcohol", "beer", "organ meats"],
    "What medications are commonly prescribed for gout?": 
        ["nsaids", "colchicine", "allopurinol", "febuxostat", "probenecid", "corticosteroids"],
    "How can gout flares be prevented?": 
        ["medication", "diet", "weight", "alcohol", "water", "exercise"],
    "Is gout related to diet?": 
        ["yes", "purine", "food", "alcohol", "seafood", "meat"],
    "Can gout be cured permanently?": 
        ["manage", "treatment", "lifestyle", "medication", "chronic"]
}

# TODO: Score the responses and calculate average scores for each strategy
# Determine which strategy performs best overall
```

## 6. Saving Results

Save your results in a structured format for auto-grading.

```python
# TODO: Save your results to results/part_3/prompting_results.txt
# The file should include:
# - Raw responses for each question and strategy
# - Scores for each question and strategy
# - Average scores for each strategy
# - The best performing strategy

# Example format:
"""
# Prompt Engineering Results

## Question: What foods should be avoided by patients with gout?

### Zero-shot response:
[response text]

### One-shot response:
[response text]

### Few-shot response:
[response text]

--------------------------------------------------

## Scores

```
question,zero_shot,one_shot,few_shot
what_foods_should,0.67,0.83,0.83
what_medications_are,0.50,0.67,0.83
how_can_gout,0.33,0.50,0.67
is_gout_related,0.80,0.80,1.00
can_gout_be,0.40,0.60,0.80

average,0.54,0.68,0.83
best_method,few_shot
```
"""
```

## Progress Checkpoints

1. **Prompting Templates**:
   - [ ] Create zero-shot template
   - [ ] Create one-shot template
   - [ ] Create few-shot template
   - [ ] Format templates with questions and examples

2. **LLM API Integration**:
   - [ ] Connect to the Hugging Face API
   - [ ] Test with different prompts
   - [ ] Handle API errors

3. **Comparison and Evaluation**:
   - [ ] Compare strategies on multiple questions
   - [ ] Score responses based on keywords
   - [ ] Determine the best strategy

4. **Results and Documentation**:
   - [ ] Save results in the required format
   - [ ] Document your findings

## What to Submit

1. Your implementation in a Python script `utils/prompt_comparison.py` that:
   - Defines the prompting templates
   - Connects to the Hugging Face API
   - Compares different prompting strategies
   - Scores and evaluates the responses

2. The results of your experiments in `results/part_3/prompting_results.txt` with the format shown above

The auto-grader will check:
1. That your results file contains the required sections
2. That your scoring logic correctly identifies keyword presence
3. That you've correctly calculated average scores
4. That you've identified the best performing method