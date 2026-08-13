import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class RAGASResult:
    """RAGAS metric results for a single sample."""
    faithfulness: float = 0.0
    relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    overall: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

class RAGASMetrics:
    """Compute RAGAS metrics for RAG evaluation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def faithfulness_score(self, answer: str, context: List[str]) -> Dict[str, Any]:
        """Faithfulness: How grounded is the answer in the context?

        Measures if claims in the answer are supported by the context.
        """
        answer_claims = self._extract_claims(answer)
        supported_claims = 0
        claim_details = []

        for claim in answer_claims:
            is_supported = self._check_claim_support(claim, context)
            if is_supported:
                supported_claims += 1
            claim_details.append({
                'claim': claim,
                'supported': is_supported,
            })

        score = supported_claims / max(len(answer_claims), 1)

        return {
            'score': float(score),
            'n_claims': len(answer_claims),
            'n_supported': supported_claims,
            'details': claim_details,
        }

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims

    def _check_claim_support(self, claim: str, context: List[str]) -> bool:
        """Check if a claim is supported by context using token overlap."""
        claim_tokens = set(claim.lower().split())

        for ctx in context:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(claim_tokens & ctx_tokens)
            overlap_ratio = overlap / max(len(claim_tokens), 1)

            if overlap_ratio > 0.3:
                return True

        return False

    def relevance_score(self, answer: str, question: str) -> Dict[str, Any]:
        """Answer Relevancy: How relevant is the answer to the question?"""
        question_tokens = set(question.lower().split())
        answer_tokens = set(answer.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'or', 'but', 'not', 'no', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
                     'other', 'some', 'such', 'than', 'too', 'very', 'just', 'that',
                     'this', 'these', 'those', 'what', 'which', 'who', 'whom', 'when',
                     'where', 'why', 'how', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                     'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their'}

        q_filtered = question_tokens - stop_words
        a_filtered = answer_tokens - stop_words

        if not q_filtered:
            return {'score': 0.5, 'overlap': 0}

        overlap = len(q_filtered & a_filtered)
        score = min(overlap / max(len(q_filtered) * 0.5, 1), 1.0)

        return {
            'score': float(score),
            'overlap': overlap,
            'question_tokens': len(q_filtered),
            'answer_tokens': len(a_filtered),
        }

    def context_precision_score(self, answer: str, context: List[str]) -> Dict[str, Any]:
        """Context Precision: How relevant is the retrieved context to the answer?"""
        if not context:
            return {'score': 0.0, 'relevant_chunks': 0}

        answer_tokens = set(answer.lower().split())

        chunk_relevances = []
        for chunk in context:
            chunk_tokens = set(chunk.lower().split())
            overlap = len(answer_tokens & chunk_tokens)
            relevance = overlap / max(len(answer_tokens), 1)
            chunk_relevances.append(relevance)

        relevant_chunks = sum(1 for r in chunk_relevances if r > 0.2)
        precision = relevant_chunks / len(context)

        # Also compute average relevance
        avg_relevance = np.mean(chunk_relevances) if chunk_relevances else 0

        return {
            'score': float(precision),
            'relevant_chunks': relevant_chunks,
            'total_chunks': len(context),
            'avg_relevance': float(avg_relevance),
            'chunk_scores': chunk_relevances,
        }

    def context_recall_score(self, context: List[str],
                            reference: str) -> Dict[str, Any]:
        """Context Recall: Does the context contain the information needed for the reference?"""
        if not context:
            return {'score': 0.0}

        ref_claims = self._extract_claims(reference)

        context_text = ' '.join(context).lower()

        recalled_claims = 0
        claim_details = []

        for claim in ref_claims:
            claim_tokens = set(claim.lower().split())
            context_tokens = set(context_text.split())

            overlap = len(claim_tokens & context_tokens)
            is_recalled = overlap / max(len(claim_tokens), 1) > 0.3

            if is_recalled:
                recalled_claims += 1

            claim_details.append({
                'claim': claim,
                'recalled': is_recalled,
                'overlap': overlap,
            })

        score = recalled_claims / max(len(ref_claims), 1)

        return {
            'score': float(score),
            'n_ref_claims': len(ref_claims),
            'n_recalled': recalled_claims,
            'details': claim_details,
        }

    def compute_all(self, answer: str, question: str,
                   context: List[str],
                   reference: str = None) -> RAGASResult:
        """Compute all RAGAS metrics."""
        faith = self.faithfulness_score(answer, context)
        relev = self.relevance_score(answer, question)
        precision = self.context_precision_score(answer, context)

        recall = {'score': 0.0}
        if reference:
            recall = self.context_recall_score(context, reference)

        scores = {
            'faithfulness': faith['score'],
            'relevance': relev['score'],
            'context_precision': precision['score'],
            'context_recall': recall['score'],
        }

        overall = np.mean(list(scores.values()))

        return RAGASResult(
            faithfulness=faith['score'],
            relevance=relev['score'],
            context_precision=precision['score'],
            context_recall=recall['score'],
            answer_relevancy=relev['score'],
            overall=float(overall),
            details={
                'faithfulness': faith,
                'relevance': relev,
                'context_precision': precision,
                'context_recall': recall,
            }
        )

    def aggregate_results(self, results: List[RAGASResult]) -> Dict[str, Any]:
        """Aggregate RAGAS results across multiple samples."""
        if not results:
            return {}

        metrics = ['faithfulness', 'relevance', 'context_precision',
                   'context_recall', 'overall']

        aggregated = {}
        for metric in metrics:
            values = [getattr(r, metric) for r in results]
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }

        aggregated['n_samples'] = len(results)

        return aggregated
