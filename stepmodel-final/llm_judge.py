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


JUDGE_SYSTEM_PROMPT = """You are an expert penetration-testing instructor evaluating the quality of step explanations in a pentesting planning system.

You will be given:
1. A predicted step explanation (what the model generated)
2. A ground truth step explanation (what a human expert wrote)
3. The predicted step (the action being explained)
4. Context from the previous step and strategy

Your task is to evaluate the predicted explanation on a scale of 1-5 for each dimension:

Evaluation Dimensions:
- Relevance (1-5): Does the explanation directly relate to and justify the predicted step?
- Accuracy (1-5): Is the technical information and reasoning correct?
- Completeness (1-5): Does it cover all important aspects of why this step is appropriate?
- Clarity (1-5): Is it well-structured, specific, and easy to understand?

Scoring Guidelines:
5 = Excellent - Perfectly meets criteria, no improvements needed
4 = Good - Meets criteria well with minor issues
3 = Adequate - Meets basic criteria but has notable flaws
2 = Poor - Fails to meet criteria in significant ways
1 = Very Poor - Completely fails to meet criteria

After scoring, provide:
1. A brief justification for each score (1-2 sentences)
2. One specific suggestion for improvement
3. An overall quality assessment (Excellent/Good/Adequate/Poor/Very Poor)

Respond in JSON format:
{
    "relevance_score": <int 1-5>,
    "relevance_justification": "<string>",
    "accuracy_score": <int 1-5>,
    "accuracy_justification": "<string>",
    "completeness_score": <int 1-5>,
    "completeness_justification": "<string>",
    "clarity_score": <int 1-5>,
    "clarity_justification": "<string>",
    "suggestion": "<string>",
    "overall_assessment": "<string>",
    "average_score": <float>
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
    user_prompt = f"""Please evaluate the following step explanation:

PREDICTED STEP: {predicted_step}

PREDICTED EXPLANATION: {predicted_explanation}

GROUND TRUTH EXPLANATION: {ground_truth_explanation}

CONTEXT:
- Previous strategy: {context.get('Previous strategy', 'N/A')}
- Previous step: {context.get('Previous step', 'N/A')}
- Previous step result: {context.get('Previous step result', 'N/A')}
- New strategy: {context.get('New strategy', 'N/A')}
- Strategy explanation: {context.get('Strategy explanation', 'N/A')}

Evaluate the predicted explanation on the dimensions specified in your instructions.
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
            "relevance_score": 3,
            "relevance_justification": "Failed to parse judge response",
            "accuracy_score": 3,
            "accuracy_justification": "Failed to parse judge response",
            "completeness_score": 3,
            "completeness_justification": "Failed to parse judge response",
            "clarity_score": 3,
            "clarity_justification": "Failed to parse judge response",
            "suggestion": "Review judge response parsing",
            "overall_assessment": "Unknown",
            "average_score": 3.0
        }, raw_response
    except Exception as e:
        print(f"[LLM Judge] Error during evaluation: {e}")
        # Return default scores on error
        return {
            "relevance_score": 3,
            "relevance_justification": f"Error: {str(e)}",
            "accuracy_score": 3,
            "accuracy_justification": f"Error: {str(e)}",
            "completeness_score": 3,
            "completeness_justification": f"Error: {str(e)}",
            "clarity_score": 3,
            "clarity_justification": f"Error: {str(e)}",
            "suggestion": "Review error logs",
            "overall_assessment": "Unknown",
            "average_score": 3.0
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
        "relevance": [],
        "accuracy": [],
        "completeness": [],
        "clarity": [],
        "average": []
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
            
            all_scores["relevance"].append(scores["relevance_score"])
            all_scores["accuracy"].append(scores["accuracy_score"])
            all_scores["completeness"].append(scores["completeness_score"])
            all_scores["clarity"].append(scores["clarity_score"])
            all_scores["average"].append(scores["average_score"])
            
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
    
    print("\nAggregate Scores (1-5 scale):")
    print("-" * 80)
    for metric, stats in aggregates.items():
        print(f"{metric.capitalize():15} | Mean: {stats['mean']:.2f} ± {stats['std']:.2f} | "
              f"Min: {stats['min']:.1f} | Max: {stats['max']:.1f} | Median: {stats['median']:.1f}")
    
    print(f"\nTotal evaluated: {results['total_evaluated']}")
    print(f"Total errors: {results['total_errors']}")
    
    # Print sample feedback
    print("\nSample Feedback:")
    print("-" * 80)
    for i, result in enumerate(results["individual_results"][:3]):
        if "scores" in result:
            print(f"\nSample {i+1} (Machine: {result['machine']}):")
            print(f"  Overall: {result['scores']['overall_assessment']}")
            print(f"  Average Score: {result['scores']['average_score']:.2f}")
            print(f"  Suggestion: {result['scores']['suggestion']}")
    
    print("=" * 80 + "\n")
