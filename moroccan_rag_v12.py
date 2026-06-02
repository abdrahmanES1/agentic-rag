# -*- coding: utf-8 -*-
"""
Backward-compatibility shim for pickled knowledge-base indexes.

The old monolith stored Chunk objects whose __module__ is 'moroccan_rag_v12'.
When Python unpickles those files it tries to import this module and look up
the class by name. Importing everything from pipeline.models satisfies that
lookup without rebuilding the index.

Do NOT delete this file as long as indexes/ contains .pkl files built before
the pipeline/ refactor. Once the index is rebuilt with the new code this
file becomes a no-op and can be removed.
"""

from pipeline.models import (  # noqa: F401 — re-exported for pickle compatibility
    Chunk,
    ScoredChunk,
    QuestionFlags,
    DocumentRecord,
    GroundingAudit,
    ClaimVerification,
    EntityVerification,
    PipelineResult,
    RetrievalResult,
    AgentState,
    AgentPlan,
    PlannedStep,
    ExecutionTrace,
    ToolCall,
)
