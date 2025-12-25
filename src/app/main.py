from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from .schemas import ChatRequest, ChatResponse, Deal
from .inference.model_loader import ModelBundle
from .inference.recommender import rerank
from .inference.text_builders import detect_missing_criteria

app = FastAPI(title="NexKey DealMatch API", version="1.0")

# Allow local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = None


@app.on_event("startup")
def load_models():
    global bundle
    bundle = ModelBundle()
    print("[STARTUP] Models loaded successfully")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "api_version": "1.0",
        "dual_vocab_size": len(bundle.dual_vocab),
        "cross_vocab_size": len(bundle.cross_vocab),
        "num_deals": len(bundle.properties),
        "checkpoints": {
            "dual_encoder": "dual_encoder_v1.pt",
            "cross_encoder": "cross_encoder_best.pt",
            "deal_vectors": "deal_vecs_v1.npy",
        },
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start_time = time.time()

    missing = detect_missing_criteria(req.message)

    # --- Clarification path ---
    if len(missing) >= 2:
        elapsed = time.time() - start_time
        print(
            f"[CHAT] {elapsed:.3f}s | clarify=True | "
            f"top_n={req.top_n} top_k={req.top_k}"
        )

        return ChatResponse(
            reply=(
                "To find the best deals, I need a bit more detail. "
                f"Can you share: {', '.join(missing)}? "
                "Example: “3 bed in AZ under 350k, entry under 20k, payment under 2500”."
            ),
            deals=[],
            needs_clarification=True,
            missing_fields=missing,
        )

    # --- Retrieve + rerank ---
    deals_df = rerank(
        bundle,
        req.message,
        top_n=req.top_n,
        top_k=req.top_k
    )

    deals = []
    for _, row in deals_df.iterrows():
        deals.append(
            Deal(
                property_id=int(row["property_id"]),
                deal_type=str(row["deal_type"]),
                city=str(row["city"]),
                state=str(row["state"]),
                beds=float(row["beds"]),
                baths=float(row["baths"]),
                sqft=float(row["sqft"]),
                purchase_price=float(row["purchase_price"]),
                arv=float(row["arv"]),
                entry_fee=float(row["entry_fee"]),
                estimated_monthly_payment=float(row["estimated_monthly_payment"]),
                rerank_score=float(row["rerank_score"]),
                retrieval_sim=float(row["retrieval_sim"]),
            )
        )

    elapsed = time.time() - start_time
    print(
        f"[CHAT] {elapsed:.3f}s | clarify=False | "
        f"top_n={req.top_n} top_k={req.top_k} | "
        f"returned={len(deals)}"
    )

    return ChatResponse(
        reply="Here are the top deals I found based on your message.",
        deals=deals,
        needs_clarification=False,
    )
