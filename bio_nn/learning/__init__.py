from .base import BaseLearningRule
from .surrogate import SurrogateGradient
from .local import LocalLearning
from .hybrid import HybridLearning

LEARNING_REGISTRY = {
    "surrogate_gradient": SurrogateGradient,
    "surrogate": SurrogateGradient,
    "local": LocalLearning,
    "hybrid": HybridLearning,
}


def create_learning_rule(rule_type, config):
    if rule_type not in LEARNING_REGISTRY:
        raise ValueError(f"Unknown learning rule: {rule_type}. Available: {list(LEARNING_REGISTRY.keys())}")
    return LEARNING_REGISTRY[rule_type](config)
