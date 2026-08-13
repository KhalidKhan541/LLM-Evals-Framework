import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class RubricCriterion:
    """Single rubric criterion for scoring."""
    name: str
    description: str
    weight: float = 1.0
    scale: tuple = (1, 5)
    examples: Dict[int, str] = field(default_factory=dict)

@dataclass
class JudgeResult:
    """Result from LLM judge scoring."""
    task_id: str
    model_name: str
    scores: Dict[str, float]
    reasoning: Dict[str, str]
    overall_score: float
    raw_response: str = None

class LLMJudge:
    """LLM-as-judge scoring with rubric prompts."""
    
    DEFAULT_RUBRIC = [
        RubricCriterion(
            name='relevance',
            description='How relevant is the response to the question?',
            weight=1.0,
            scale=(1, 5),
            examples={
                1: 'Completely off-topic or unrelated',
                3: 'Partially relevant but misses key points',
                5: 'Directly addresses the question fully',
            }
        ),
        RubricCriterion(
            name='accuracy',
            description='How factually accurate is the response?',
            weight=1.5,
            scale=(1, 5),
            examples={
                1: 'Contains major factual errors',
                3: 'Mostly accurate with minor errors',
                5: 'Completely accurate with verifiable facts',
            }
        ),
        RubricCriterion(
            name='completeness',
            description='How complete is the response?',
            weight=1.0,
            scale=(1, 5),
            examples={
                1: 'Missing most required information',
                3: 'Covers main points but missing details',
                5: 'Comprehensive coverage of all aspects',
            }
        ),
        RubricCriterion(
            name='clarity',
            description='How clear and well-structured is the response?',
            weight=0.8,
            scale=(1, 5),
            examples={
                1: 'Confusing and poorly organized',
                3: 'Understandable but could be clearer',
                5: 'Crystal clear and well-organized',
            }
        ),
    ]
    
    def __init__(self, model_name: str = 'gpt-4',
                 custom_rubric: List[RubricCriterion] = None):
        self.model_name = model_name
        self.rubric = custom_rubric or self.DEFAULT_RUBRIC
        self.logger = logging.getLogger(__name__)
    
    def build_judge_prompt(self, input_text: str, output: str,
                          expected: str = None,
                          context: List[str] = None,
                          criteria: List[str] = None) -> str:
        """Build the judge prompt with rubric."""
        relevant_criteria = self.rubric
        if criteria:
            relevant_criteria = [c for c in self.rubric if c.name in criteria]
        
        rubric_text = "\n\n".join([
            f"### {c.name.upper()} (weight: {c.weight})\n"
            f"{c.description}\n"
            f"Scale: {c.scale[0]}-{c.scale[1]}\n"
            + "\n".join([f"- {score}: {desc}" for score, desc in c.examples.items()])
            for c in relevant_criteria
        ])
        
        context_text = ""
        if context:
            context_text = "\n\nContext:\n" + "\n".join([f"- {c}" for c in context])
        
        expected_text = ""
        if expected:
            expected_text = f"\n\nExpected Output:\n{expected}"
        
        prompt = f"""You are an expert evaluator scoring LLM outputs.

## Task
Input: {input_text}{context_text}{expected_text}

## Model Output
{output}

## Scoring Rubric
{rubric_text}

## Instructions
1. Score each criterion on the scale provided
2. Provide brief reasoning for each score
3. Calculate weighted overall score

Respond in JSON format:
{{
    "scores": {{
        "criterion_name": score,
        ...
    }},
    "reasoning": {{
        "criterion_name": "brief explanation",
        ...
    }},
    "overall_score": weighted_average
}}
"""
        return prompt
    
    def parse_judge_response(self, response: str,
                            task_id: str = None,
                            model_name: str = None) -> JudgeResult:
        """Parse LLM judge response into structured result."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                self.logger.error(f"Failed to parse judge response")
                return JudgeResult(
                    task_id=task_id,
                    model_name=model_name,
                    scores={},
                    reasoning={},
                    overall_score=0.0,
                    raw_response=response,
                )
        
        scores = data.get('scores', {})
        reasoning = data.get('reasoning', {})
        
        # Calculate weighted overall if not provided
        overall = data.get('overall_score')
        if overall is None:
            total_weight = sum(c.weight for c in self.rubric if c.name in scores)
            if total_weight > 0:
                overall = sum(
                    scores.get(c.name, 0) * c.weight
                    for c in self.rubric if c.name in scores
                ) / total_weight
            else:
                overall = sum(scores.values()) / max(len(scores), 1)
        
        return JudgeResult(
            task_id=task_id,
            model_name=model_name,
            scores=scores,
            reasoning=reasoning,
            overall_score=float(overall),
            raw_response=response,
        )
    
    def pairwise_judge_prompt(self, input_text: str,
                              output_a: str, output_b: str,
                              context: List[str] = None) -> str:
        """Build pairwise comparison judge prompt."""
        context_text = ""
        if context:
            context_text = "\n\nContext:\n" + "\n".join([f"- {c}" for c in context])
        
        return f"""Compare two model outputs for the given input.

## Input
{input_text}{context_text}

## Output A
{output_a}

## Output B
{output_b}

## Evaluation Criteria
1. Accuracy: Which is more factually correct?
2. Relevance: Which better addresses the question?
3. Completeness: Which provides more useful information?
4. Clarity: Which is better written and organized?

Respond in JSON:
{{
    "winner": "A" or "B" or "tie",
    "scores": {{"A": score_1_to_5, "B": score_1_to_5}},
    "reasoning": "explanation of comparison"
}}
"""
    
    def scoring_prompt_with_examples(self, input_text: str, output: str,
                                    criterion: RubricCriterion) -> str:
        """Build scoring prompt with few-shot examples."""
        examples_text = "\n".join([
            f"Score {score}: {desc}"
            for score, desc in criterion.examples.items()
        ])
        
        return f"""Score the following output on: {criterion.name}

Criterion: {criterion.description}

Examples:
{examples_text}

Input: {input_text}
Output: {output}

Provide your score (1-{criterion.scale[1]}) and brief reasoning in JSON:
{{"score": X, "reasoning": "explanation"}}
"""
