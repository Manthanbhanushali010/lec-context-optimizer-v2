"""LEC Context Optimizer — FastAPI Service"""

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from optimizer.optimizer import ContextOptimizer
from optimizer.assembler import Assembler
from optimizer.types import Message, MessageRole, QueryType

app = FastAPI(title="LEC Context Optimizer", version="1.0.0")
_optimizer = ContextOptimizer()
_assembler = Assembler()


class MessageIn(BaseModel):
    role: str
    content: str


class OptimizeRequest(BaseModel):
    messages: list[MessageIn] = Field(..., min_length=1)
    query: str = Field(..., min_length=3)
    query_type: str | None = None


@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    messages = [
        Message(index=i, role=MessageRole(m.role), content=m.content)
        for i, m in enumerate(req.messages)
    ]
    qt = QueryType(req.query_type) if req.query_type else None
    try:
        result = _optimizer.optimize(messages, req.query, qt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    thread = _assembler.to_anthropic_messages(result)
    return {
        "optimized_messages": thread,
        "original_token_count": result.original_token_count,
        "optimized_token_count": result.optimized_token_count,
        "token_reduction_pct": result.token_reduction_pct,
        "query_type_detected": result.query_type.value,
        "landmarks_preserved": result.landmarks_preserved,
        "compressed_groups": result.compressed_groups,
        "assembly_latency_ms": result.assembly_latency_ms,
        "compression_cost_usd": result.compression_cost_usd,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "lec-context-optimizer"}
