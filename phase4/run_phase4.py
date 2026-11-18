"""
CASTER Phase 4 -- CLI runner.

Starts the FastAPI backend server, and prints instructions for launching
the React frontend. Mirrors the structure of run_phase1.py, run_phase2.py,
and run_phase3.py so all four phases feel like one continuous pipeline.

Usage:
    python phase4/run_phase4.py                  # start backend on port 8000
    python phase4/run_phase4.py --port 8080      # custom port
    python phase4/run_phase4.py --check          # just check imports, no server

Full pipeline recap:
    python phase1/run_phase1.py --input data/raw/clip.mp4
    python phase2/run_phase2.py --clip clip --sport cricket --classifier xgboost
    python phase3/run_phase3.py --clip clip
    python phase4/run_phase4.py                  # then open http://localhost:5173
"""
import argparse
import sys
import os
from pathlib import Path

# Fix Windows console UTF-8 output encoding for checkmarks / unicode symbols
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parent.parent))


def check_imports():
    """Verify that all Phase 1-4 imports resolve before starting the server."""
    print("Checking Phase 4 imports...\n")
    checks = [
        ("config",                            "shared config"),
        ("schemas",                           "shared schemas"),
        ("phase2.scene_json_builder",         "Phase 2 scene JSON builder"),
        ("phase2.confidence_gate",            "Phase 2 confidence gate"),
        ("phase3.inference.commentary_engine","Phase 3 commentary engine"),
        ("phase3.inference.template_fallback","Phase 3 template fallback"),
        ("phase4.backend.main",               "Phase 4 FastAPI app"),
        ("phase4.backend.pipeline_runner",    "Phase 4 pipeline runner"),
        ("phase4.backend.ws_handler",         "Phase 4 WebSocket handler"),
    ]

    all_ok = True
    for module, label in checks:
        try:
            __import__(module)
            print(f"  ✓  {label:<40} ({module})")
        except ImportError as e:
            print(f"  ✗  {label:<40} IMPORT ERROR: {e}")
            all_ok = False

    # Optional heavy imports (Phase 1 CV stack)
    optional = [
        ("ultralytics", "YOLOv8 (Phase 1 detection)"),
        ("cv2",         "OpenCV (Phase 1 video I/O)"),
        ("xgboost",     "XGBoost (Phase 2 classifier)"),
        ("groq",        "Groq API client (Phase 3 backend)"),
    ]
    print()
    for module, label in optional:
        try:
            __import__(module)
            print(f"  ✓  {label:<40} ({module})")
        except ImportError:
            print(f"  ⚠  {label:<40} NOT INSTALLED — optional, see requirements.txt")

    print()
    if all_ok:
        print("All required imports OK. Phase 4 is ready to run.")
    else:
        print("Some required imports failed. Check the errors above.")
        sys.exit(1)

    # GROQ_API_KEY hint
    if not os.environ.get("GROQ_API_KEY"):
        print()
        print("  ℹ  GROQ_API_KEY not set — commentary will use template fallback.")
        print("     Set it for better quality:  export GROQ_API_KEY=<your_key>")
        print("     Free signup at: https://console.groq.com")


def main():
    parser = argparse.ArgumentParser(description="CASTER Phase 4 — Start the full pipeline server")
    parser.add_argument("--port",  type=int, default=8000, help="FastAPI server port (default 8000)")
    parser.add_argument("--host",  default="0.0.0.0",     help="FastAPI bind host (default 0.0.0.0)")
    parser.add_argument("--check", action="store_true",    help="Check imports only, do not start server")
    parser.add_argument("--reload",action="store_true",    help="Enable uvicorn --reload (dev mode)")
    args = parser.parse_args()

    check_imports()

    if args.check:
        return

    print("\n" + "=" * 60)
    print("CASTER Phase 4 — Starting backend server")
    print("=" * 60)
    print(f"  Backend:   http://{args.host}:{args.port}")
    print(f"  API docs:  http://localhost:{args.port}/docs")
    print()
    print("  To start the React frontend (in a separate terminal):")
    print("    cd phase4/frontend")
    print("    npm install")
    print("    npm run dev")
    print("  Then open: http://localhost:5173")
    print()
    print("  Full pipeline:")
    print("    1. python phase1/run_phase1.py --input data/raw/clip.mp4")
    print("    2. python phase2/run_phase2.py --clip clip --sport cricket --classifier xgboost")
    print("    3. python phase3/run_phase3.py --clip clip")
    print("    4. python phase4/run_phase4.py  (this command)")
    print("    5. Upload the same clip at http://localhost:5173")
    print("=" * 60 + "\n")

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    # Check if target port is already in use (common Windows error 10048)
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((args.host if args.host != "0.0.0.0" else "127.0.0.1", args.port)) == 0:
            print(f"⚠️  WARNING: Port {args.port} is already in use by another process.")
            alt_port = args.port + 1
            while True:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_alt:
                    if s_alt.connect_ex(("127.0.0.1", alt_port)) != 0:
                        break
                alt_port += 1
            print(f"   Switching automatically to available port: {alt_port}")
            print(f"   (Or kill the old process using: Stop-Process -Id <PID> -Force)\n")
            args.port = alt_port

    uvicorn.run(
        "phase4.backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
