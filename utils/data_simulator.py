"""
Synthetic network traffic generation for IoT-IDS demonstration.

This module generates realistic IoT network traffic patterns for simulation
and testing of the intrusion detection system.
"""

import random
from typing import Tuple, List, Optional
from enum import Enum

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# PCA components (PC1-PC16)
N_FEATURES = 16

# Threat types (matching actual model classes)
class ThreatType(str, Enum):
    """Enumeration of IoT threat types."""
    NORMAL = "normal"
    BRUTE_FORCE = "brute_force"
    DDOS = "ddos"
    MITM = "mitm"
    SCAN = "scan"
    SPOOFING = "spoofing"

# Severity levels for visual alerts
THREAT_SEVERITY = {
    ThreatType.NORMAL: "normal",
    ThreatType.SCAN: "low",
    ThreatType.SPOOFING: "medium",
    ThreatType.BRUTE_FORCE: "high",
    ThreatType.MITM: "high",
    ThreatType.DDOS: "critical",
}

# =============================================================================
# ATTACK PATTERN DEFINITIONS
# =============================================================================

# Statistical patterns for each attack type (based on PCA characteristics)
ATTACK_PATTERNS = {
    ThreatType.NORMAL: {
        "mean": np.array([0.0] * N_FEATURES),
        "std": np.array([1.0] * N_FEATURES),
        "skew": None
    },
    ThreatType.DDOS: {
        "mean": np.array([3.0, 2.5, 2.0, 1.5, 1.0] + [0.0] * 11),
        "std": np.array([0.8, 0.7, 0.6, 0.5, 0.4] + [1.0] * 11),
        "skew": [0, 1, 2]  # Indices with positive skew
    },
    ThreatType.BRUTE_FORCE: {
        "mean": np.array([0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 2.5] + [0.0] * 9),
        "std": np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.4, 0.3] + [1.0] * 9),
        "skew": [4, 5, 6]
    },
    ThreatType.SPOOFING: {
        "mean": np.array([0.0, 0.0, 2.0, 1.5, 1.0, 0.5] + [0.0] * 10),
        "std": np.array([1.0, 1.0, 0.6, 0.5, 0.4, 0.3] + [1.0] * 10),
        "skew": [2, 3]
    },
    ThreatType.MITM: {
        "mean": np.array([0.5, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0] + [0.0] * 9),
        "std": np.array([1.2, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3] + [1.0] * 9),
        "skew": [3, 4, 5]
    },
    ThreatType.SCAN: {
        "mean": np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5] + [0.0] * 7),
        "std": np.array([0.7, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.4] + [1.0] * 7),
        "skew": [7, 8]
    },
}

# =============================================================================
# CORE GENERATION FUNCTIONS
# =============================================================================

def generate_traffic_sample(threat_type: Optional[ThreatType] = None) -> Tuple[np.ndarray, str]:
    """
    Generate a single synthetic traffic sample.

    Args:
        threat_type: Specific threat type to generate, or None for random

    Returns:
        Tuple of (sample_array, threat_label)

    Example:
        >>> sample, label = generate_traffic_sample(ThreatType.DDOS)
        >>> print(f"Generated {label} sample with shape {sample.shape}")
    """
    # Select threat type
    if threat_type is None:
        threat_type = random.choice(list(ThreatType))
    elif isinstance(threat_type, str):
        threat_type = ThreatType(threat_type)

    # Get pattern parameters
    pattern = ATTACK_PATTERNS[threat_type]
    mean = pattern["mean"]
    std = pattern["std"]
    skew_indices = pattern.get("skew")

    # Generate base sample
    sample = np.random.normal(mean, std)

    # Apply skewness to specific components
    if skew_indices:
        for idx in skew_indices:
            # Add positive skew using exponential distribution
            sample[idx] += np.abs(np.random.exponential(0.5))

    # Add small random noise for realism
    noise = np.random.normal(0, 0.1, N_FEATURES)
    sample += noise

    return sample, threat_type.value

def generate_attack_burst(
    threat_type: ThreatType,
    count: int = 10,
    variability: float = 0.3
) -> List[Tuple[np.ndarray, str]]:
    """
    Generate a burst of related attack samples.

    Useful for simulating sustained attacks or scanning activities.

    Args:
        threat_type: Type of attack to generate
        count: Number of samples in burst
        variability: Amount of variation between samples (0-1)

    Returns:
        List of (sample, label) tuples

    Example:
        >>> burst = generate_attack_burst(ThreatType.DDOS, count=20)
        >>> print(f"Generated {len(burst)} DDoS samples")
    """
    samples = []

    # Generate base template
    base_sample, label = generate_traffic_sample(threat_type)

    for _ in range(count):
        # Add controlled variation
        variation = np.random.normal(0, variability, N_FEATURES)
        sample = base_sample + variation
        samples.append((sample, label))

    return samples

def generate_mixed_traffic(
    duration_seconds: int = 60,
    samples_per_second: int = 1,
    threat_ratio: float = 0.3
) -> List[Tuple[float, np.ndarray, str]]:
    """
    Generate realistic mixed traffic over time.

    Args:
        duration_seconds: Duration of simulation
        samples_per_second: Sampling rate
        threat_ratio: Proportion of malicious traffic (0-1)

    Returns:
        List of (timestamp, sample, label) tuples

    Example:
        >>> traffic = generate_mixed_traffic(duration_seconds=30)
        >>> print(f"Generated {len(traffic)} samples over 30 seconds")
    """
    total_samples = duration_seconds * samples_per_second
    num_threats = int(total_samples * threat_ratio)
    num_benign = total_samples - num_threats

    timeline = []

    for i in range(total_samples):
        timestamp = i / samples_per_second

        # Decide if this sample is threat or normal
        if i < num_threats:
            # Generate random threat
            threat_types = [t for t in ThreatType if t != ThreatType.NORMAL]
            threat_type = random.choice(threat_types)
            sample, label = generate_traffic_sample(threat_type)
        else:
            # Generate normal traffic
            sample, label = generate_traffic_sample(ThreatType.NORMAL)

        timeline.append((timestamp, sample, label))

    # Shuffle to make realistic
    random.shuffle(timeline)

    # Re-sort by timestamp
    timeline.sort(key=lambda x: x[0])

    return timeline

def generate_scenario_traffic(
    scenario: str,
    duration: int = 60
) -> List[Tuple[float, np.ndarray, str]]:
    """
    Generate traffic for specific attack scenarios.

    Args:
        scenario: Scenario name ('normal', 'under_attack', 'scanning', 'mixed')
        duration: Duration in seconds

    Returns:
        List of (timestamp, sample, label) tuples

    Example:
        >>> traffic = generate_scenario_traffic('under_attack', duration=30)
    """
    if scenario == "normal":
        # Mostly normal with very few threats
        return generate_mixed_traffic(duration, threat_ratio=0.05)

    elif scenario == "under_attack":
        # Heavy DDoS attack
        timeline = []
        for i in range(duration):
            timestamp = float(i)
            # 80% attack traffic
            if random.random() < 0.8:
                sample, label = generate_traffic_sample(ThreatType.DDOS)
            else:
                sample, label = generate_traffic_sample(ThreatType.NORMAL)
            timeline.append((timestamp, sample, label))
        return timeline

    elif scenario == "scanning":
        # Port scanning activity
        timeline = []
        for i in range(duration):
            timestamp = float(i)
            if random.random() < 0.6:
                sample, label = generate_traffic_sample(ThreatType.SCAN)
            else:
                sample, label = generate_traffic_sample(ThreatType.NORMAL)
            timeline.append((timestamp, sample, label))
        return timeline

    else:  # mixed or default
        return generate_mixed_traffic(duration, threat_ratio=0.3)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_threat_severity(threat_type: str) -> str:
    """
    Get severity level for a threat type.

    Args:
        threat_type: Name of threat type

    Returns:
        Severity level: 'normal', 'low', 'medium', 'high', 'critical'
    """
    try:
        threat_enum = ThreatType(threat_type)
        return THREAT_SEVERITY[threat_enum]
    except (ValueError, KeyError):
        return "low"

def get_all_threat_types() -> List[str]:
    """Get list of all threat type names."""
    return [t.value for t in ThreatType]

def get_attack_types_only() -> List[str]:
    """Get list of attack types (excluding normal)."""
    return [t.value for t in ThreatType if t != ThreatType.NORMAL]

def calculate_risk_score(predictions: List[str]) -> float:
    """
    Calculate overall risk score based on recent predictions.

    Args:
        predictions: List of recent prediction labels

    Returns:
        Risk score (0-100)
    """
    if not predictions:
        return 0.0

    severity_weights = {
        "normal": 0,
        "low": 15,
        "medium": 40,
        "high": 70,
        "critical": 100
    }

    total_weight = 0
    for pred in predictions:
        severity = get_threat_severity(pred)
        total_weight += severity_weights.get(severity, 0)

    # Average with exponential weighting (recent samples matter more)
    weights = np.exp(np.linspace(0, 1, len(predictions)))
    weighted_avg = total_weight / len(predictions)

    return min(weighted_avg, 100.0)
