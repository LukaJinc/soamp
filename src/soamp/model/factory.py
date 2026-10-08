"""Thin factory tying a built Featurization (from soamp.data.factory) to a
constructed model, so a notebook or pipeline/train.py never has to derive
peptide_feature_dim/organism_vocab_size (or feature_dim, for a vector-based
organism strategy) by hand.

`architecture` selects which model class gets built -- mirrors the same
plain-string-selector + `**kwargs` registry pattern already used for
peptide/organism featurization (see
soamp.features.{peptide,organism}_featurizers.build_*_featurizer).
"""
from typing import TYPE_CHECKING, Literal

from soamp.model.attention_fusion import AttentionFusionClassifier
from soamp.model.baseline_mlp import BaselineClassifier
from soamp.model.graph_encoder import GINEncoder, GraphPeptideClassifier

if TYPE_CHECKING:
    from soamp.data.factory import DatasetBundle


class ModelFactoryError(ValueError):
    """Raised when neither dataset_bundle nor the dims required by the
    active organism_output_kind are given, or when an unknown architecture
    is requested."""


def build_model(
    dataset_bundle: "DatasetBundle | None" = None,
    *,
    architecture: str = "baseline_classifier",
    peptide_feature_dim: int | None = None,
    organism_output_kind: Literal["index", "vector"] = "index",
    organism_vocab_size: int | None = None,
    organism_feature_dim: int | None = None,
    peptide_input_kind: Literal["vector", "graph"] = "vector",
    graph_node_dim: int | None = None,
    graph_edge_dim: int | None = None,
    graph_encoder_kwargs: dict | None = None,
    **architecture_kwargs,
):
    """For peptide_input_kind="graph" the chosen architecture is built on top
    of a GINEncoder (kwargs from `graph_encoder_kwargs`): the encoder turns a
    graph batch into a fixed-length vector of size `out_dim`, which the
    architecture then treats exactly like a descriptor/embedding vector."""
    if dataset_bundle is not None:
        peptide_input_kind = dataset_bundle.featurization.peptide_input_kind
        graph_node_dim = dataset_bundle.featurization.graph_node_dim
        graph_edge_dim = dataset_bundle.featurization.graph_edge_dim
        peptide_feature_dim = dataset_bundle.featurization.peptide_feature_dim
        organism_output_kind = dataset_bundle.featurization.organism_output_kind
        organism_vocab_size = dataset_bundle.featurization.organism_vocab_size
        organism_feature_dim = dataset_bundle.featurization.organism_feature_dim

    peptide_encoder = None
    if peptide_input_kind == "graph":
        if graph_node_dim is None or graph_edge_dim is None:
            raise ModelFactoryError(
                "peptide_input_kind='graph' needs graph_node_dim and graph_edge_dim "
                "(or a dataset_bundle)"
            )
        peptide_encoder = GINEncoder(graph_node_dim, graph_edge_dim, **(graph_encoder_kwargs or {}))
        peptide_feature_dim = peptide_encoder.out_dim
    elif peptide_input_kind != "vector":
        raise ModelFactoryError(f"unknown peptide_input_kind: {peptide_input_kind!r}")

    missing_dims = (
        peptide_feature_dim is None
        or (organism_output_kind == "index" and organism_vocab_size is None)
        or (organism_output_kind == "vector" and organism_feature_dim is None)
    )
    if missing_dims:
        raise ModelFactoryError(
            "pass dataset_bundle, or peptide_feature_dim plus organism_vocab_size "
            "(output_kind='index') or organism_feature_dim (output_kind='vector')"
        )

    common_kwargs = dict(
        peptide_feature_dim=peptide_feature_dim,
        organism_output_kind=organism_output_kind,
        organism_vocab_size=organism_vocab_size,
        organism_feature_dim=organism_feature_dim,
    )
    if architecture == "baseline_classifier":
        classifier = BaselineClassifier(**common_kwargs, **architecture_kwargs)
    elif architecture == "attention_fusion_classifier":
        classifier = AttentionFusionClassifier(**common_kwargs, **architecture_kwargs)
    else:
        raise ModelFactoryError(f"unknown architecture: {architecture!r}")
    if peptide_encoder is not None:
        return GraphPeptideClassifier(peptide_encoder, classifier)
    return classifier
