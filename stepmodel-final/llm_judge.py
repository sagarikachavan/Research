"""
llm_judge.py — LLM-based evaluation of step explanations.

Uses a powerful LLM (GPT-4 or similar) to evaluate the quality of predicted
step explanations by comparing them to ground truth explanations.

The judge evaluates on multiple dimensions:
- Relevance: Does the explanation relate to the predicted step?
- Accuracy: Is the technical information correct?
- Completeness: Does it cover all important aspects?
- Clarity: Is it well-structured and easy to understand?
- Overall quality combined score

Usage:
    from llm_judge import evaluate_explanation_with_llm
    score, feedback = evaluate_explanation_with_llm(
        predicted_explanation="...",
        ground_truth_explanation="...",
        predicted_step="...",
        context={...}
    )
"""
import json
import os
from typing import Dict, Any, Tuple
import openai


JUDGE_SYSTEM_PROMPT = """You are an expert penetration-testing instructor evaluating student answers in a pentesting planning system.

You will be given:
1. A predicted step explanation (what the model/student generated)
2. A ground truth step explanation (what a human expert wrote)
3. The predicted step (the action being explained)
4. Context from the previous step and strategy

Your task is to evaluate whether the predicted explanation conveys the SAME MEANING as the ground truth explanation, like a teacher grading a student's answer.

Evaluation Criteria:
- Does the predicted explanation convey the same core reasoning and justification as the ground truth?
- Are the technical concepts and logic equivalent, even if worded differently?
- Would this explanation be acceptable as a correct answer in a classroom setting?

Scoring:
- Return a correctness score between 0.0 and 1.0
- 1.0 = Perfect match - conveys exactly the same meaning and reasoning
- 0.8-0.9 = Very good - minor differences in wording but same core meaning
- 0.6-0.7 = Good - mostly correct with some minor omissions or slight inaccuracies
- 0.4-0.5 = Partial - captures some key points but misses important aspects
- 0.2-0.3 = Poor - misses the main point or has significant errors
- 0.0-0.1 = Very poor - completely wrong or irrelevant

Respond in JSON format:
{
    "correctness_score": <float 0.0-1.0>,
    "justification": "<brief explanation of why this score was given>",
    "is_correct": <boolean - true if score >= 0.6, false otherwise>
}"""


def evaluate_explanation_with_llm(
    predicted_explanation: str,
    ground_truth_explanation: str,
    predicted_step: str,
    context: Dict[str, Any],
    model: str = "gpt-4o",
    api_key: str = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluate a predicted explanation using an LLM judge.
    
    Args:
        predicted_explanation: The model-generated explanation
        ground_truth_explanation: The human-written ground truth
        predicted_step: The step being explained
        context: Dictionary with context fields (previous step, strategy, etc.)
        model: OpenAI model to use for judging
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    Returns:
        Tuple of (scores_dict, raw_response)
    """
    if api_key:
        openai.api_key = api_key
    elif not openai.api_key:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai.api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
    
    # Build the evaluation prompt
    user_prompt = f"""Please evaluate whether the predicted explanation conveys the same meaning as the ground truth explanation:

PREDICTED STEP: {predicted_step}

PREDICTED EXPLANATION: {predicted_explanation}

GROUND TRUTH EXPLANATION: {ground_truth_explanation}

CONTEXT:
- Previous strategy: {context.get('Previous strategy', 'N/A')}
- Previous step: {context.get('Previous step', 'N/A')}
- Previous step result: {context.get('Previous step result', 'N/A')}
- New strategy: {context.get('New strategy', 'N/A')}
- Strategy explanation: {context.get('Strategy explanation', 'N/A')}

Evaluate whether the predicted explanation is correct and conveys the same meaning as the ground truth.
Provide your response in JSON format as requested."""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent scoring
            max_tokens=1000,
        )
        
        raw_response = response.choices[0].message.content
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0].strip()
        else:
            json_str = raw_response.strip()
        
        scores = json.loads(json_str)
        
        return scores, raw_response
        
    except json.JSONDecodeError as e:
        print(f"[LLM Judge] Failed to parse JSON response: {e}")
        print(f"[LLM Judge] Raw response: {raw_response}")
        # Return default scores if parsing fails
        return {
            "correctness_score": 0.0,
            "justification": "Failed to parse judge response",
            "is_correct": False
        }, raw_response
    except Exception as e:
        print(f"[LLM Judge] Error during evaluation: {e}")
        # Return default scores on error
        return {
            "correctness_score": 0.0,
            "justification": f"Error: {str(e)}",
            "is_correct": False
        }, str(e)


def batch_evaluate_explanations(
    examples: list,
    model: str = "gpt-4o",
    api_key: str = None,
    max_samples: int = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate multiple explanations using LLM judge.
    
    Args:
        examples: List of example dicts with predicted_explanation, 
                  gold_step_explanation, step_label, context, etc.
        model: OpenAI model to use
        api_key: OpenAI API key
        max_samples: Maximum number of samples to evaluate (for testing)
        verbose: Print progress updates
    
    Returns:
        Dictionary with aggregated scores and individual results
    """
    if max_samples:
        examples = examples[:max_samples]
    
    results = []
    all_scores = {
        "correctness": [],
        "is_correct": []
    }
    
    total = len(examples)
    for i, ex in enumerate(examples, 1):
        if verbose:
            print(f"[LLM Judge] Evaluating {i}/{total}...")
        
        try:
            scores, raw = evaluate_explanation_with_llm(
                predicted_explanation=ex.get("pred_explanation", ""),
                ground_truth_explanation=ex.get("gold_explanation", ""),
                predicted_step=ex.get("pred_step", ex.get("gold_step", "")),
                context=ex.get("context", {}),
                model=model,
                api_key=api_key,
            )
            
            results.append({
                "index": i,
                "machine": ex.get("machine", ""),
                "scores": scores,
                "raw_response": raw
            })
            
            all_scores["correctness"].append(scores["correctness_score"])
            all_scores["is_correct"].append(scores["is_correct"])
            
        except Exception as e:
            print(f"[LLM Judge] Error evaluating sample {i}: {e}")
            results.append({
                "index": i,
                "machine": ex.get("machine", ""),
                "error": str(e)
            })
    
    # Calculate aggregates
    import numpy as np
    
    aggregates = {}
    for key, values in all_scores.items():
        if values:
            if key == "is_correct":
                # Calculate accuracy percentage for boolean values
                accuracy = sum(values) / len(values) * 100
                aggregates[key] = {
                    "accuracy_percent": float(accuracy),
                    "correct_count": int(sum(values)),
                    "total_count": len(values)
                }
            else:
                aggregates[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }
    
    return {
        "aggregates": aggregates,
        "individual_results": results,
        "total_evaluated": len(results),
        "total_errors": sum(1 for r in results if "error" in r)
    }


def print_llm_judge_results(results: Dict[str, Any]):
    """Print formatted LLM judge results."""
    print("\n" + "=" * 80)
    print("LLM JUDGE EVALUATION RESULTS")
    print("=" * 80)
    
    aggregates = results["aggregates"]
    
    print("\nAggregate Scores:")
    print("-" * 80)
    
    # Print correctness statistics
    if "correctness" in aggregates:
        stats = aggregates["correctness"]
        print(f"Correctness Score | Mean: {stats['mean']:.3f} ± {stats['std']:.3f} | "
              f"Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Median: {stats['median']:.3f}")
    
    # Print accuracy percentage
    if "is_correct" in aggregates:
        stats = aggregates["is_correct"]
        print(f"\nAccuracy: {stats['accuracy_percent']:.2f}% ({stats['correct_count']}/{stats['total_count']} correct)")
    
    print(f"\nTotal evaluated: {results['total_evaluated']}")
    print(f"Total errors: {results['total_errors']}")
    
    # Print sample feedback
    print("\nSample Feedback:")
    print("-" * 80)
    for i, result in enumerate(results["individual_results"][:3]):
        if "scores" in result:
            print(f"\nSample {i+1} (Machine: {result['machine']}):")
            print(f"  Correctness: {result['scores']['correctness_score']:.3f}")
            print(f"  Is Correct: {result['scores']['is_correct']}")
            print(f"  Justification: {result['scores']['justification']}")
    
    print("=" * 80 + "\n")
