"""
CASTER Phase 3 -- Commentary Engine. FOOTBALL ONLY.

Phase 3 reads outputs/<clip>_scene_json_events.json produced by
phase2/run_phase2.py and feeds each scene JSON into the LLM to generate
a commentary sentence.

Actual interface:

    from phase3.inference.commentary_engine import CommentaryEngine

    scene_json: dict   # one item from *_scene_json_events.json
    engine = CommentaryEngine()
    result = engine.generate(scene_json)
    # result["text"] -- one sentence of spoken commentary

Phase 3 components:
    phase3/
        inference/
            commentary_engine.py -- backend-fallback orchestrator (local_gguf -> groq -> template)
            groq_client.py       -- Groq free-tier API backend
            template_fallback.py -- zero-dependency rule-based backend, always succeeds
            prompt_builder.py    -- prompt template builder, football scene-JSON shape
        data_collection/
            scrape_bbc_sport.py       -- BBC Sport live text scraper
            format_training_pairs.py  -- formats (scene_json, commentary) training pairs
        finetune/
            qlora_finetune.py    -- QLoRA fine-tuning on Colab T4
            train_style_adapters.py, merge_adapter.py, convert_to_gguf.sh
        backend/
            main.py, websocket_stream.py, redis_queue.py -- FastAPI/WebSocket layer
        run_phase3.py            -- orchestrator: reads Phase 2 JSON, generates commentary

Groq free tier (effective primary: openai/gpt-oss-120b, see
.env's GROQ_MODEL) is the recommended immediate-use path before
QLoRA fine-tuning completes; local_gguf activates automatically once
CASTER_GGUF_MODEL_PATH points at a real fine-tuned model.
"""
