"""
Advanced Cytogenetics Parser for Myeloid Leukemia Risk Stratification

This module implements a comprehensive ISCN (International System for Human Cytogenomic Nomenclature)
parser specifically designed for myeloid leukemia prognosis. It extracts prognostically significant
chromosomal abnormalities and converts them into quantitative risk features.

"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class CytogeneticRisk(Enum):
    """ELN 2022 cytogenetic risk categories"""
    FAVORABLE = 0
    INTERMEDIATE = 1
    ADVERSE = 2


@dataclass
class CytogeneticFeatures:
    """Container for extracted cytogenetic features"""
    n_abnormalities: int = 0
    is_complex_karyotype: bool = False
    has_favorable_abnormality: bool = False
    has_adverse_abnormality: bool = False
    risk_score: float = 1.0  # 0=favorable, 1=intermediate, 2=adverse

    # Specific abnormality flags
    has_t_8_21: bool = False
    has_inv_16: bool = False
    has_t_16_16: bool = False
    has_monosomy_7: bool = False
    has_del_5q: bool = False
    has_del_7q: bool = False
    has_t_9_22: bool = False
    has_inv_3: bool = False
    has_t_3_3: bool = False
    has_t_6_9: bool = False

    # Additional features
    karyotype_complexity: int = 0
    has_normal_karyotype: bool = False
    has_hyperdiploidy: bool = False
    has_hypodiploidy: bool = False


class LeukemiaCytogeneticsParser:
    """
    Advanced ISCN cytogenetics parser for myeloid leukemia risk stratification.

    This parser handles the complex ISCN nomenclature used in leukemia cytogenetics
    and extracts features that are prognostic according to ELN 2022 guidelines.
    """

    def __init__(self):
        # Initialize pattern dictionaries
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for different cytogenetic abnormalities"""

        # Favorable abnormalities (ELN 2022)
        self.favorable_patterns = {
            't_8_21': re.compile(r't\(8;21\)'),
            'inv_16': re.compile(r'inv\(16\)'),
            't_16_16': re.compile(r't\(16;16\)'),
            't_15_17': re.compile(r't\(15;17\)'),  # APL specific
        }

        # Adverse abnormalities (ELN 2022)
        self.adverse_patterns = {
            'inv_3': re.compile(r'inv\(3\)'),
            't_3_3': re.compile(r't\(3;3\)'),
            't_6_9': re.compile(r't\(6;9\)'),
            't_9_22': re.compile(r't\(9;22\)'),
            'monosomy_7': re.compile(r'-7'),
            'del_5q': re.compile(r'del\(5q\)'),
            'del_7q': re.compile(r'del\(7q\)'),
            'monosomy_5': re.compile(r'-5'),
        }

        # General patterns
        self.general_patterns = {
            'translocation': re.compile(r't\(\d+;\d+\)'),
            'inversion': re.compile(r'inv\(\d+\)'),
            'deletion': re.compile(r'del\(\d+[qQpP]\)'),
            'duplication': re.compile(r'dup\(\d+[qQpP]\)'),
            'addition': re.compile(r'add\(\d+[qQpP]\)'),
            'monosomy': re.compile(r'-\d+'),
            'trisomy': re.compile(r'\+\d+'),
            'marker_chromosome': re.compile(r'\+\w+'),
            'ring_chromosome': re.compile(r'r\(\d+\)'),
            'dicentric': re.compile(r'dic\(\d+;\d+\)'),
        }

        # Normal karyotype pattern
        self.normal_pattern = re.compile(r'^46,\w{2}$')

        # Complex karyotype indicators
        self.complex_indicators = [
            re.compile(r'\d{2,},\w+'),  # High chromosome count
            re.compile(r'^\d{2,}'),     # Starting with high number
        ]

    def parse_cytogenetics(self, karyotype_string: Optional[str]) -> CytogeneticFeatures:
        """
        Parse ISCN karyotype string and extract prognostic features.

        Args:
            karyotype_string: ISCN formatted karyotype string

        Returns:
            CytogeneticFeatures object with extracted features
        """

        if pd.isna(karyotype_string) or karyotype_string == '' or karyotype_string.lower() == 'normal':
            return self._get_normal_karyotype_features()

        # Clean and standardize the karyotype string
        karyotype = self._clean_karyotype_string(karyotype_string)

        features = CytogeneticFeatures()

        # Check for normal karyotype
        if self.normal_pattern.match(karyotype):
            features.has_normal_karyotype = True
            return features

        # Count total abnormalities
        features.n_abnormalities = self._count_abnormalities(karyotype)

        # Check for complex karyotype (≥3 abnormalities)
        features.is_complex_karyotype = features.n_abnormalities >= 3

        # Extract specific abnormalities
        features = self._extract_specific_abnormalities(karyotype, features)

        # Calculate karyotype complexity
        features.karyotype_complexity = self._calculate_complexity(karyotype)

        # Determine risk category
        features = self._determine_risk_category(features)

        return features

    def _clean_karyotype_string(self, karyotype: str) -> str:
        """Clean and standardize karyotype string"""

        # Remove extra whitespace and normalize
        karyotype = karyotype.strip()

        # Convert to lowercase for consistent processing
        karyotype = karyotype.lower()

        # Remove common artifacts
        karyotype = re.sub(r'\[.*?\]', '', karyotype)  # Remove clone information
        karyotype = re.sub(r'cp\d*', '', karyotype)    # Remove clone prefix
        karyotype = re.sub(r'sl', '', karyotype)       # Remove slide information

        return karyotype.strip()

    def _count_abnormalities(self, karyotype: str) -> int:
        """Count distinct chromosomal abnormalities"""

        abnormality_count = 0

        # Count each type of abnormality
        for pattern_name, pattern in self.general_patterns.items():
            matches = pattern.findall(karyotype)
            abnormality_count += len(matches)

        # Special handling for complex abnormalities
        if ',' in karyotype:
            # Split by commas and count structural abnormalities
            parts = karyotype.split(',')
            for part in parts[1:]:  # Skip chromosome count
                if any(char in part for char in ['t', 'inv', 'del', 'dup', 'add', '-', '+']):
                    abnormality_count += 1

        return abnormality_count

    def _extract_specific_abnormalities(self, karyotype: str,
                                      features: CytogeneticFeatures) -> CytogeneticFeatures:
        """Extract specific prognostically significant abnormalities"""

        # Check favorable abnormalities
        for abbrev, pattern in self.favorable_patterns.items():
            if pattern.search(karyotype):
                features.has_favorable_abnormality = True
                setattr(features, f'has_{abbrev.replace("_", "_")}', True)

        # Check adverse abnormalities
        for abbrev, pattern in self.adverse_patterns.items():
            if pattern.search(karyotype):
                features.has_adverse_abnormality = True
                setattr(features, f'has_{abbrev}', True)

        # Check for ploidy abnormalities
        if self._has_hyperdiploidy(karyotype):
            features.has_hyperdiploidy = True
        if self._has_hypodiploidy(karyotype):
            features.has_hypodiploidy = True

        return features

    def _has_hyperdiploidy(self, karyotype: str) -> bool:
        """Check for hyperdiploid karyotype (>46 chromosomes)"""
        match = re.match(r'(\d+)', karyotype)
        if match:
            chromosome_count = int(match.group(1))
            return chromosome_count > 46
        return False

    def _has_hypodiploidy(self, karyotype: str) -> bool:
        """Check for hypodiploid karyotype (<46 chromosomes)"""
        match = re.match(r'(\d+)', karyotype)
        if match:
            chromosome_count = int(match.group(1))
            return chromosome_count < 46
        return False

    def _calculate_complexity(self, karyotype: str) -> int:
        """Calculate karyotype complexity score"""

        complexity = 0

        # Base complexity from abnormality count
        abnormality_count = self._count_abnormalities(karyotype)
        complexity += abnormality_count

        # Additional complexity for specific patterns
        if re.search(r'mar|\+', karyotype):  # Marker chromosomes
            complexity += 1
        if re.search(r'r\(|dic\(', karyotype):  # Rings or dicentrics
            complexity += 2
        if re.search(r'ins\(|hsr', karyotype):  # Insertions or homogeneously staining regions
            complexity += 2

        # Complexity bonus for multiple chromosome involvement
        chromosomes_involved = set()
        for match in re.finditer(r'\((\d+);?', karyotype):
            chromosomes_involved.add(match.group(1))

        if len(chromosomes_involved) > 3:
            complexity += 1

        return complexity

    def _determine_risk_category(self, features: CytogeneticFeatures) -> CytogeneticFeatures:
        """Determine ELN 2022 risk category"""

        # Adverse abnormalities take precedence
        if features.has_adverse_abnormality or features.is_complex_karyotype:
            features.risk_score = CytogeneticRisk.ADVERSE.value
        elif features.has_favorable_abnormality:
            features.risk_score = CytogeneticRisk.FAVORABLE.value
        else:
            features.risk_score = CytogeneticRisk.INTERMEDIATE.value

        # Special cases
        if features.has_normal_karyotype:
            features.risk_score = CytogeneticRisk.INTERMEDIATE.value

        return features

    def _get_normal_karyotype_features(self) -> CytogeneticFeatures:
        """Return features for normal karyotype"""
        features = CytogeneticFeatures()
        features.has_normal_karyotype = True
        features.risk_score = CytogeneticRisk.INTERMEDIATE.value
        return features

    def features_to_dataframe(self, karyotype_series: pd.Series) -> pd.DataFrame:
        """
        Convert a pandas Series of karyotype strings to feature DataFrame

        Args:
            karyotype_series: Pandas Series containing karyotype strings

        Returns:
            DataFrame with extracted features as columns
        """

        feature_dicts = []

        for karyotype in karyotype_series:
            features = self.parse_cytogenetics(karyotype)
            feature_dict = {
                'n_cytogenetic_abnormalities': features.n_abnormalities,
                'is_complex_karyotype': features.is_complex_karyotype,
                'has_favorable_abnormality': features.has_favorable_abnormality,
                'has_adverse_abnormality': features.has_adverse_abnormality,
                'cytogenetic_risk_score': features.risk_score,
                'karyotype_complexity': features.karyotype_complexity,
                'has_normal_karyotype': features.has_normal_karyotype,
                'has_hyperdiploidy': features.has_hyperdiploidy,
                'has_hypodiploidy': features.has_hypodiploidy,
                # Specific abnormalities
                'has_t_8_21': features.has_t_8_21,
                'has_inv_16': features.has_inv_16,
                'has_t_16_16': features.has_t_16_16,
                'has_monosomy_7': features.has_monosomy_7,
                'has_del_5q': features.has_del_5q,
                'has_del_7q': features.has_del_7q,
                'has_t_9_22': features.has_t_9_22,
                'has_inv_3': features.has_inv_3,
                'has_t_3_3': features.has_t_3_3,
                'has_t_6_9': features.has_t_6_9,
            }
            feature_dicts.append(feature_dict)

        return pd.DataFrame(feature_dicts)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return estimated feature importance weights for risk prediction"""

        return {
            'cytogenetic_risk_score': 0.25,
            'is_complex_karyotype': 0.20,
            'has_adverse_abnormality': 0.15,
            'has_favorable_abnormality': 0.15,
            'n_cytogenetic_abnormalities': 0.10,
            'karyotype_complexity': 0.08,
            'has_normal_karyotype': 0.05,
            'has_monosomy_7': 0.02,
            'has_del_5q': 0.02,
            'has_t_8_21': 0.02,
            'has_inv_16': 0.02,
            'has_hyperdiploidy': 0.01,
            'has_hypodiploidy': 0.01,
            'has_del_7q': 0.01,
            'has_t_9_22': 0.01,
            'has_inv_3': 0.01,
            'has_t_3_3': 0.01,
            'has_t_6_9': 0.01,
            'has_t_16_16': 0.01,
        }


def create_cytogenetic_features(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add cytogenetic features to clinical dataframe

    Args:
        clinical_df: DataFrame with CYTOGENETICS column

    Returns:
        DataFrame with additional cytogenetic features
    """

    if 'CYTOGENETICS' not in clinical_df.columns:
        raise ValueError("Clinical dataframe must contain 'CYTOGENETICS' column")

    parser = LeukemiaCytogeneticsParser()

    # Extract cytogenetic features
    cyto_features = parser.features_to_dataframe(clinical_df['CYTOGENETICS'])

    # Combine with original clinical data
    enhanced_clinical = pd.concat([clinical_df, cyto_features], axis=1)

    return enhanced_clinical


# Example usage and testing
if __name__ == "__main__":

    # Example karyotypes
    test_karyotypes = [
        "46,XX",  # Normal female
        "46,XY",  # Normal male
        "45,XY,-7",  # Monosomy 7
        "46,XY,t(8;21)(q22;q22)",  # Favorable translocation
        "46,XY,inv(16)(p13q22)",  # Favorable inversion
        "44,XY,-5,-7,del(12p),+mar",  # Complex karyotype
        "47,XX,+8",  # Trisomy 8
        "43,XY,-5,-7,-17,-18,+mar1,+mar2",  # Very complex
    ]

    parser = LeukemiaCytogeneticsParser()

    print("Cytogenetics Parser Testing Results:")
    print("=" * 50)

    for karyotype in test_karyotypes:
        features = parser.parse_cytogenetics(karyotype)
        print(f"\nKaryotype: {karyotype}")
        print(f"  Abnormalities: {features.n_abnormalities}")
        print(f"  Complex: {features.is_complex_karyotype}")
        print(f"  Risk Score: {features.risk_score}")
        print(f"  Normal: {features.has_normal_karyotype}")
        if features.has_t_8_21:
            print("  ✓ Has t(8;21)")
        if features.has_inv_16:
            print("  ✓ Has inv(16)")
        if features.has_monosomy_7:
            print("  ✓ Has monosomy 7")
