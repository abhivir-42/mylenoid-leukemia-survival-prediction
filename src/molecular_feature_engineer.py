"""
Advanced Molecular Feature Engineering for Myeloid Leukemia Survival Prediction

This module implements comprehensive molecular feature engineering for leukemia prognosis,
transforming raw mutation data into prognostic biomarkers that capture gene-gene interactions,
pathway alterations, and clonal architecture.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum


class MutationEffect(Enum):
    """Categorization of mutation effects"""
    LOSS_OF_FUNCTION = "loss_of_function"
    GAIN_OF_FUNCTION = "gain_of_function"
    DOMINANT_NEGATIVE = "dominant_negative"
    UNKNOWN = "unknown"


@dataclass
class GeneAnnotation:
    """Gene annotation with prognostic information"""
    gene: str
    pathway: str
    prognostic_impact: str  # favorable, adverse, neutral
    mutation_effect: MutationEffect
    typical_vaf_range: Tuple[float, float]


class LeukemiaMolecularFeatureEngineer:
    """
    Advanced molecular feature engineering for leukemia survival prediction.

    This class transforms long-format molecular data into comprehensive feature sets
    that capture mutation patterns, pathway alterations, and gene-gene interactions.
    """

    def __init__(self):
        self._init_gene_annotations()
        self._init_pathway_definitions()
        self._init_interaction_rules()

    def _init_gene_annotations(self):
        """Initialize gene annotations with prognostic information"""

        self.gene_annotations = {
            # Adverse prognostic genes
            'TP53': GeneAnnotation('TP53', 'tumor_suppressor', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.9)),
            'FLT3': GeneAnnotation('FLT3', 'signaling', 'adverse', MutationEffect.GAIN_OF_FUNCTION, (0.2, 0.8)),
            'ASXL1': GeneAnnotation('ASXL1', 'epigenetic', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.7)),
            'RUNX1': GeneAnnotation('RUNX1', 'transcription', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.2, 0.8)),
            'EZH2': GeneAnnotation('EZH2', 'epigenetic', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.6)),
            'IDH1': GeneAnnotation('IDH1', 'epigenetic', 'adverse', MutationEffect.GAIN_OF_FUNCTION, (0.2, 0.6)),
            'IDH2': GeneAnnotation('IDH2', 'epigenetic', 'adverse', MutationEffect.GAIN_OF_FUNCTION, (0.2, 0.6)),
            'SRSF2': GeneAnnotation('SRSF2', 'splicing', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.7)),
            'STAG2': GeneAnnotation('STAG2', 'cohesin', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.4, 0.8)),
            'BCOR': GeneAnnotation('BCOR', 'epigenetic', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.7)),

            # Favorable prognostic genes
            'NPM1': GeneAnnotation('NPM1', 'nucleolus', 'favorable', MutationEffect.DOMINANT_NEGATIVE, (0.3, 0.7)),
            'CEBPA': GeneAnnotation('CEBPA', 'transcription', 'favorable', MutationEffect.LOSS_OF_FUNCTION, (0.2, 0.6)),

            # Signaling pathway genes
            'KIT': GeneAnnotation('KIT', 'signaling', 'adverse', MutationEffect.GAIN_OF_FUNCTION, (0.2, 0.6)),
            'RAS': GeneAnnotation('RAS', 'signaling', 'adverse', MutationEffect.GAIN_OF_FUNCTION, (0.1, 0.5)),
            'PTPN11': GeneAnnotation('PTPN11', 'signaling', 'adverse', MutationEffect.GAIN_OF_FUNCTION, (0.2, 0.6)),
            'NF1': GeneAnnotation('NF1', 'signaling', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.7)),

            # Epigenetic regulators
            'DNMT3A': GeneAnnotation('DNMT3A', 'epigenetic', 'neutral', MutationEffect.LOSS_OF_FUNCTION, (0.1, 0.5)),
            'TET2': GeneAnnotation('TET2', 'epigenetic', 'neutral', MutationEffect.LOSS_OF_FUNCTION, (0.2, 0.6)),
            'ASXL1': GeneAnnotation('ASXL1', 'epigenetic', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.7)),
            'EZH2': GeneAnnotation('EZH2', 'epigenetic', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.6)),

            # Transcription factors
            'RUNX1': GeneAnnotation('RUNX1', 'transcription', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.2, 0.8)),
            'CEBPA': GeneAnnotation('CEBPA', 'transcription', 'favorable', MutationEffect.LOSS_OF_FUNCTION, (0.2, 0.6)),
            'GATA2': GeneAnnotation('GATA2', 'transcription', 'adverse', MutationEffect.LOSS_OF_FUNCTION, (0.3, 0.7)),
            'ETV6': GeneAnnotation('ETV6', 'transcription', 'neutral', MutationEffect.LOSS_OF_FUNCTION, (0.2, 0.6)),
        }

    def _init_pathway_definitions(self):
        """Initialize pathway definitions for feature engineering"""

        self.pathway_definitions = {
            'epigenetic': ['DNMT3A', 'TET2', 'IDH1', 'IDH2', 'ASXL1', 'EZH2', 'BCOR'],
            'signaling': ['FLT3', 'KIT', 'RAS', 'PTPN11', 'NF1', 'JAK2'],
            'transcription': ['RUNX1', 'CEBPA', 'GATA2', 'ETV6', 'TP53'],
            'splicing': ['SRSF2', 'SF3B1', 'U2AF1'],
            'cohesin': ['STAG2', 'RAD21', 'SMC1A', 'SMC3'],
            'tumor_suppressor': ['TP53', 'WT1', 'PHF6', 'NF1'],
        }

    def _init_interaction_rules(self):
        """Initialize gene-gene interaction rules"""

        self.interaction_rules = {
            # NPM1 prognostic impact modified by FLT3
            'npm1_flt3_interaction': {
                'genes': ['NPM1', 'FLT3'],
                'rule': lambda npm1, flt3: 'favorable' if npm1 and not flt3 else 'adverse' if npm1 and flt3 else 'neutral'
            },

            # TP53 high VAF indicates poor prognosis
            'tp53_high_vaf': {
                'genes': ['TP53'],
                'rule': lambda tp53_vaf: tp53_vaf > 0.4 if tp53_vaf > 0 else False
            },

            # Epigenetic mutations often co-occur
            'epigenetic_module': {
                'genes': ['DNMT3A', 'TET2', 'IDH1', 'IDH2'],
                'rule': lambda mutations: len([m for m in mutations if m]) >= 2
            }
        }

    def create_molecular_features(self, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform long-format molecular data into comprehensive feature matrix

        Args:
            molecular_df: DataFrame with columns [ID, CHR, START, END, REF, ALT, GENE, PROTEIN_CHANGE, EFFECT, VAF, DEPTH]

        Returns:
            DataFrame with patient-level molecular features
        """

        print(f"Processing {len(molecular_df)} mutations for {molecular_df['ID'].nunique()} patients...")

        # Create base patient-level features
        patient_features = self._create_base_features(molecular_df)

        # Add mutation burden features
        patient_features = self._add_mutation_burden_features(patient_features, molecular_df)

        # Add pathway-level features
        patient_features = self._add_pathway_features(patient_features, molecular_df)

        # Add gene-specific features
        patient_features = self._add_gene_specific_features(patient_features, molecular_df)

        # Add interaction features
        patient_features = self._add_interaction_features(patient_features, molecular_df)

        # Add VAF-based features
        patient_features = self._add_vaf_features(patient_features, molecular_df)

        # Add clonal architecture features
        patient_features = self._add_clonal_features(patient_features, molecular_df)

        print(f"Created {len(patient_features.columns)} molecular features for {len(patient_features)} patients")

        return patient_features

    def _create_base_features(self, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Create base patient-level features from molecular data"""

        # Group by patient
        patient_groups = molecular_df.groupby('ID')

        # Basic mutation counts per gene
        gene_counts = pd.pivot_table(
            molecular_df,
            values='VAF',
            index='ID',
            columns='GENE',
            aggfunc='count',
            fill_value=0
        ).add_prefix('mut_').astype(int)

        # Mean VAF per gene
        gene_vaf = pd.pivot_table(
            molecular_df,
            values='VAF',
            index='ID',
            columns='GENE',
            aggfunc='mean',
            fill_value=0
        ).add_prefix('vaf_')

        # Effect type indicators
        effect_features = self._create_effect_features(molecular_df)

        # Combine features
        base_features = pd.concat([gene_counts, gene_vaf, effect_features], axis=1)

        return base_features

    def _create_effect_features(self, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on mutation effects"""

        effect_pivot = pd.pivot_table(
            molecular_df,
            values='VAF',
            index='ID',
            columns=['GENE', 'EFFECT'],
            aggfunc='count',
            fill_value=0
        )

        # Flatten column names
        effect_pivot.columns = [f"{gene}_{effect}_count" for gene, effect in effect_pivot.columns]
        effect_pivot = effect_pivot.reset_index()

        return effect_pivot.set_index('ID')

    def _add_mutation_burden_features(self, features: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Add mutation burden and diversity features"""

        patient_stats = molecular_df.groupby('ID').agg({
            'GENE': 'count',  # Total mutations
            'VAF': ['mean', 'std', 'max', 'min'],
            'EFFECT': lambda x: len(x.unique()),  # Effect diversity
        }).round(4)

        # Flatten column names
        patient_stats.columns = [
            'total_mutations',
            'mean_vaf', 'std_vaf', 'max_vaf', 'min_vaf',
            'effect_diversity'
        ]

        # Add derived features
        patient_stats['mutation_density'] = patient_stats['total_mutations'] / patient_stats['total_mutations'].max()
        patient_stats['vaf_range'] = patient_stats['max_vaf'] - patient_stats['min_vaf']
        patient_stats['high_vaf_fraction'] = (molecular_df[molecular_df['VAF'] > 0.5]
                                            .groupby('ID')['VAF'].count() /
                                            molecular_df.groupby('ID')['VAF'].count()).fillna(0)

        # Merge with features
        features = features.merge(patient_stats, left_index=True, right_index=True, how='left')

        return features.fillna(0)

    def _add_pathway_features(self, features: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Add pathway-level mutation features"""

        pathway_features = pd.DataFrame(index=features.index)

        for pathway_name, genes in self.pathway_definitions.items():
            # Count mutations in pathway
            pathway_cols = [col for col in features.columns
                          if col.startswith('mut_') and col[4:] in genes]

            if pathway_cols:
                pathway_features[f'{pathway_name}_mutations'] = features[pathway_cols].sum(axis=1)
                pathway_features[f'{pathway_name}_burden'] = features[pathway_cols].sum(axis=1) / len(genes)

                # Mean VAF in pathway
                vaf_cols = [col.replace('mut_', 'vaf_') for col in pathway_cols if col.replace('mut_', 'vaf_') in features.columns]
                if vaf_cols:
                    pathway_features[f'{pathway_name}_mean_vaf'] = features[vaf_cols].mean(axis=1)

        # Merge pathway features
        features = pd.concat([features, pathway_features], axis=1)

        return features

    def _add_gene_specific_features(self, features: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Add gene-specific prognostic features"""

        gene_features = pd.DataFrame(index=features.index)

        # Key prognostic genes
        key_genes = ['TP53', 'FLT3', 'NPM1', 'DNMT3A', 'TET2', 'IDH1', 'IDH2']

        for gene in key_genes:
            mut_col = f'mut_{gene}'
            vaf_col = f'vaf_{gene}'

            if mut_col in features.columns:
                gene_features[f'has_{gene}_mutation'] = (features[mut_col] > 0).astype(int)

                if vaf_col in features.columns:
                    gene_features[f'{gene}_high_vaf'] = ((features[mut_col] > 0) &
                                                        (features[vaf_col] > 0.4)).astype(int)

        # Special features for key genes
        if 'mut_TP53' in features.columns and 'vaf_TP53' in features.columns:
            gene_features['tp53_adverse_profile'] = ((features['mut_TP53'] > 0) &
                                                   (features['vaf_TP53'] > 0.4)).astype(int)

        if all(col in features.columns for col in ['mut_NPM1', 'mut_FLT3']):
            gene_features['npm1_favorable'] = ((features['mut_NPM1'] > 0) &
                                             (features['mut_FLT3'] == 0)).astype(int)

        features = pd.concat([features, gene_features], axis=1)

        return features

    def _add_interaction_features(self, features: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Add gene-gene interaction features"""

        interaction_features = pd.DataFrame(index=features.index)

        # NPM1-FLT3 interaction
        if 'mut_NPM1' in features.columns and 'mut_FLT3' in features.columns:
            npm1_present = features['mut_NPM1'] > 0
            flt3_present = features['mut_FLT3'] > 0

            interaction_features['npm1_without_flt3'] = ((npm1_present) & (~flt3_present)).astype(int)
            interaction_features['npm1_with_flt3'] = ((npm1_present) & (flt3_present)).astype(int)

        # Epigenetic module mutations
        epigenetic_genes = ['DNMT3A', 'TET2', 'IDH1', 'IDH2']
        epigenetic_cols = [f'mut_{gene}' for gene in epigenetic_genes if f'mut_{gene}' in features.columns]

        if epigenetic_cols:
            epigenetic_count = features[epigenetic_cols].sum(axis=1)
            interaction_features['epigenetic_module'] = (epigenetic_count >= 2).astype(int)

        # Multiple signaling mutations
        signaling_genes = ['FLT3', 'KIT', 'RAS', 'PTPN11']
        signaling_cols = [f'mut_{gene}' for gene in signaling_genes if f'mut_{gene}' in features.columns]

        if signaling_cols:
            signaling_count = features[signaling_cols].sum(axis=1)
            interaction_features['multiple_signaling'] = (signaling_count >= 2).astype(int)

        features = pd.concat([features, interaction_features], axis=1)

        return features

    def _add_vaf_features(self, features: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Add VAF-based features"""

        vaf_features = pd.DataFrame(index=features.index)

        # VAF statistics per patient
        patient_vaf_stats = molecular_df.groupby('ID')['VAF'].agg([
            'mean', 'std', 'max', 'min', 'median'
        ]).add_prefix('patient_vaf_')

        # VAF distribution features
        vaf_features['vaf_skewness'] = self._calculate_vaf_skewness(molecular_df)
        vaf_features['high_vaf_mutations'] = molecular_df[molecular_df['VAF'] > 0.5].groupby('ID')['VAF'].count()
        vaf_features['low_vaf_mutations'] = molecular_df[molecular_df['VAF'] < 0.2].groupby('ID')['VAF'].count()

        # Driver gene VAF features
        driver_genes = ['TP53', 'FLT3', 'NPM1', 'DNMT3A']
        for gene in driver_genes:
            if f'vaf_{gene}' in features.columns:
                vaf_features[f'{gene}_vaf_category'] = pd.cut(
                    features[f'vaf_{gene}'],
                    bins=[0, 0.2, 0.4, 0.6, 1.0],
                    labels=['low', 'medium', 'high', 'very_high']
                ).astype(str).replace('nan', 'no_mutation')

        # Combine all VAF features
        all_vaf_features = pd.concat([patient_vaf_stats, vaf_features], axis=1)
        features = pd.concat([features, all_vaf_features], axis=1)

        return features.fillna(0)

    def _add_clonal_features(self, features: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Add clonal architecture features"""

        clonal_features = pd.DataFrame(index=features.index)

        # Clonal vs subclonal mutations
        clonal_features['clonal_mutations'] = molecular_df[molecular_df['VAF'] > 0.4].groupby('ID')['VAF'].count()
        clonal_features['subclonal_mutations'] = molecular_df[molecular_df['VAF'] <= 0.4].groupby('ID')['VAF'].count()

        # Clonality index
        total_mutations = molecular_df.groupby('ID')['VAF'].count()
        clonal_features['clonality_index'] = (clonal_features['clonal_mutations'] /
                                            total_mutations).fillna(0)

        # Multiple clones indicator
        clonal_features['multiple_clones'] = (clonal_features['subclonal_mutations'] > 0).astype(int)

        features = pd.concat([features, clonal_features], axis=1)

        return features.fillna(0)

    def _calculate_vaf_skewness(self, molecular_df: pd.DataFrame) -> pd.Series:
        """Calculate VAF distribution skewness per patient"""

        def skewness(series):
            if len(series) < 3:
                return 0
            return series.skew()

        return molecular_df.groupby('ID')['VAF'].agg(skewness)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return estimated feature importance weights"""

        return {
            # High importance features
            'total_mutations': 0.08,
            'epigenetic_mutations': 0.07,
            'signaling_mutations': 0.07,
            'tp53_adverse_profile': 0.06,
            'npm1_favorable': 0.06,
            'max_vaf': 0.05,
            'clonality_index': 0.05,

            # Medium importance features
            'epigenetic_module': 0.04,
            'multiple_signaling': 0.04,
            'high_vaf_fraction': 0.03,
            'effect_diversity': 0.03,

            # Lower importance features
            'transcription_mutations': 0.02,
            'splicing_mutations': 0.02,
            'cohesin_mutations': 0.02,
            'tumor_suppressor_mutations': 0.02,
        }


def create_molecular_features(molecular_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create molecular features

    Args:
        molecular_df: Raw molecular data DataFrame

    Returns:
        Processed molecular features DataFrame
    """

    engineer = LeukemiaMolecularFeatureEngineer()
    return engineer.create_molecular_features(molecular_df)


# Example usage and testing
if __name__ == "__main__":

    # Create sample molecular data for testing
    sample_data = pd.DataFrame({
        'ID': ['P001', 'P001', 'P001', 'P002', 'P002', 'P003', 'P003', 'P003', 'P003'],
        'GENE': ['TP53', 'FLT3', 'DNMT3A', 'NPM1', 'DNMT3A', 'TP53', 'FLT3', 'NPM1', 'ASXL1'],
        'VAF': [0.6, 0.3, 0.2, 0.4, 0.3, 0.8, 0.2, 0.5, 0.4],
        'EFFECT': ['nonsense', 'missense', 'missense', 'frameshift', 'missense',
                  'nonsense', 'missense', 'frameshift', 'nonsense'],
        'PROTEIN_CHANGE': ['p.R175H', 'p.D835Y', 'p.R882H', 'p.W288fs', 'p.R882C',
                          'p.R273C', 'p.D835H', 'p.L287fs', 'p.G645fs'],
        'CHR': ['17', '13', '2', '5', '2', '17', '13', '5', '20'],
        'START': [7577539, 28608258, 25457242, 170837543, 25457242,
                 7578413, 28608258, 170837543, 31022441],
        'END': [7577539, 28608258, 25457242, 170837543, 25457242,
               7578413, 28608258, 170837543, 31022441],
        'REF': ['C', 'G', 'C', 'A', 'C', 'C', 'G', 'T', 'A'],
        'ALT': ['T', 'T', 'T', 'G', 'G', 'A', 'A', 'C', 'G']
    })

    print("Testing Molecular Feature Engineer...")
    print("=" * 50)

    engineer = LeukemiaMolecularFeatureEngineer()
    features = engineer.create_molecular_features(sample_data)

    print(f"\nGenerated {len(features)} patients with {len(features.columns)} features")
    print("\nSample features for first patient:")
    print(features.iloc[0].head(20))

    print("\nFeature importance ranking:")
    importance = engineer.get_feature_importance()
    for feature, weight in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(".3f")
