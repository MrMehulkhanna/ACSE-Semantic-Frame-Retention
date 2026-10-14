"""
ACSE — Module 6: SHAP Trace Explainer
========================================

For every retained frame, produces a human-readable explanation of WHY it was
retained — which factor dominated and by how much.  Uses Shapley-style marginal
contribution analysis.

Related approaches:
  Zhuang 1998, Liu & Cheng 2008, and Zhang et al. 2016 provide no per-frame
  explanations.  VA-RED2 explains computation reduction as a ratio but not
  individual-frame decisions.  ReTaKe lacks frame-level attribution.
  Our explainer computes exact marginal contributions of each factor via
  leave-one-out re-fusion and generates human-readable reason strings, making
  every retention decision auditable.
"""


class SHAPTraceExplainer:
    """Marginal-contribution explainer for retention decisions."""

    FACTOR_NAMES = {
        "cni": "Cognitive Novelty Index",
        "epd": "Event Persistence Decay",
        "sss": "Semantic Surprise Score",
        "pig": "Predictive Information Gain",
        "raw": "Resource-Aware Weight",
    }

    def __init__(self, fusion_engine) -> None:
        """
        Args:
            fusion_engine: A MultiFactorFusion instance with fuse_without().
        """
        self.fusion_engine = fusion_engine

    def explain_decision(self, fusion_output: dict, frame_idx: int) -> dict:
        """Compute Shapley-style marginal attributions for this frame."""
        full_score = fusion_output["importance_score"]
        factors = ["cni", "epd", "sss", "pig", "raw"]

        attribution = {}
        for f in factors:
            score_without = self.fusion_engine.fuse_without(f, fusion_output)
            attribution[f] = round(full_score - score_without, 6)

        # Dominant factor = highest absolute marginal contribution
        dominant = max(attribution, key=lambda k: abs(attribution[k]))

        decision = fusion_output.get("decision", "RETAIN" if full_score >= 0.5 else "DISCARD")

        human_reason = self.generate_human_reason(attribution, fusion_output)

        return {
            "frame_idx": frame_idx,
            "decision": decision,
            "importance_score": round(full_score, 4),
            "dominant_factor": dominant,
            "attribution": attribution,
            "human_reason": human_reason,
        }

    def generate_human_reason(self, attribution: dict, fusion_output: dict) -> str:
        """Template-based human-readable explanation."""
        # Sort by absolute contribution
        ranked = sorted(attribution.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top2 = ranked[:2]

        total_abs = sum(abs(v) for _, v in attribution.items())
        if total_abs < 1e-9:
            return "All factors contributed equally; score near neutral."

        parts = []
        for factor, marginal in top2:
            val = fusion_output.get(factor, 0.0)
            pct = abs(marginal) / total_abs * 100
            level = self._level_word(val, factor)
            name = self.FACTOR_NAMES.get(factor, factor.upper())
            parts.append(
                f"{level} {name.lower()} ({factor.upper()}={val:.2f}), "
                f"contributing {pct:.0f}% of retention decision"
            )

        return "; ".join(parts) + "."

    @staticmethod
    def _level_word(val: float, factor: str) -> str:
        """Quantise a factor value to high/moderate/low."""
        if factor == "epd":
            # EPD is a penalty, so high val = high event persistence
            if val > 0.6:
                return "High"
            if val > 0.25:
                return "Moderate"
            return "Low"
        else:
            if val > 0.7:
                return "High"
            if val > 0.35:
                return "Moderate"
            return "Low"
