"""
Opens a local miniplayer in your browser: the processed clip plays with
every frame's detections (ball/player/goalkeeper/referee) drawn on top in
sync, using the same colors as inspect_detections.py. Standard video
controls -- play, pause, scrub, speed -- let you watch tracking quality
over a whole stretch instead of checking one static frame at a time.

Reuses outputs/<clip>_scene_states.json from a prior run_phase1.py run --
run that first if you haven't.

Usage (from project root):
    python phase1/run_phase1.py --input data/raw/Final.mp4 --start 255 --max-frames 300
    python phase1/view_tracking.py --video data/raw/Final.mp4

--clip defaults to --video's filename stem (e.g. "Final" from
"data/raw/Final.mp4"), matching what run_phase1.py would have named its
output. Pass --clip explicitly only if your saved outputs file uses a
different stem than the video file -- if the names don't look related, you
get a warning rather than a silent, meaningless overlay (wrong clip's boxes
drawn on the wrong video).

Serves the video + detection data over a local HTTP server (needed so the
browser can load the JSON -- a bare file:// page can't) and opens your
default browser automatically. Close the terminal / Ctrl+C to stop serving.
"""
import argparse
import http.server
import json
import socket
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLORS = {   # matches inspect_detections.py's palette
    "ball": "#ff3b30",
    "player": "#30d158",
    "goalkeeper": "#ff9f0a",
    "referee": "#ffd60a",
}

PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CASTER -- tracking viewer: {clip}</title>
<style>
  body {{ background:#111; color:#eee; font-family:sans-serif; margin:0; padding:20px; }}
  #wrap {{ position:relative; display:inline-block; }}
  video {{ display:block; max-width:100%; }}
  canvas {{ position:absolute; top:0; left:0; pointer-events:none; }}
  #info {{ margin-top:10px; font-size:14px; line-height:1.6; white-space:pre; }}
  .legend span {{ margin-right:16px; }}
  .sw {{ display:inline-block; width:10px; height:10px; margin-right:4px; vertical-align:middle; }}
</style></head>
<body>
<h3>{clip} -- tracking viewer</h3>
<div class="legend">
  <span><span class="sw" style="background:{ball}"></span>ball</span>
  <span><span class="sw" style="background:{player}"></span>player</span>
  <span><span class="sw" style="background:{goalkeeper}"></span>goalkeeper</span>
  <span><span class="sw" style="background:{referee}"></span>referee</span>
</div>
<div id="wrap">
  <video id="v" src="/video" controls></video>
  <canvas id="c"></canvas>
</div>
<div id="info">loading detection data...</div>
<script>
const COLORS = {colors_json};
let scenes = [];
let t0 = null, t1 = null;

fetch('/data').then(r => r.json()).then(data => {{
  scenes = data;
  t0 = scenes[0].timestamp_sec;
  t1 = scenes[scenes.length - 1].timestamp_sec;
  document.getElementById('info').textContent =
    `${{scenes.length}} processed frames, ${{t0.toFixed(1)}}s - ${{t1.toFixed(1)}}s. ` +
    `Play/scrub the video above -- boxes outside that range are not available (shown blank).`;
}});

function nearestScene(t) {{
  if (!scenes.length) return null;
  let lo = 0, hi = scenes.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (scenes[mid].timestamp_sec < t) lo = mid + 1; else hi = mid;
  }}
  // lo is first index >= t; compare against lo-1 too for nearest
  if (lo > 0 && Math.abs(scenes[lo - 1].timestamp_sec - t) < Math.abs(scenes[lo].timestamp_sec - t)) lo -= 1;
  return scenes[lo];
}}

const v = document.getElementById('v');
const c = document.getElementById('c');
const ctx = c.getContext('2d');

function resize() {{
  c.width = v.clientWidth;
  c.height = v.clientHeight;
}}
window.addEventListener('resize', resize);
v.addEventListener('loadedmetadata', resize);

function draw() {{
  requestAnimationFrame(draw);
  if (!scenes.length || !v.videoWidth) return;
  ctx.clearRect(0, 0, c.width, c.height);
  const sx = c.width / v.videoWidth, sy = c.height / v.videoHeight;
  const t = v.currentTime;
  const status = document.getElementById('info');

  if (t < t0 || t > t1) {{
    status.textContent = `t=${{t.toFixed(2)}}s -- outside processed range (${{t0.toFixed(1)}}-${{t1.toFixed(1)}}s), no data`;
    return;
  }}
  const s = nearestScene(t);
  let lines = [`t=${{t.toFixed(2)}}s (nearest processed frame ${{s.timestamp_sec.toFixed(2)}}s)`,
               `ball_detected=${{s.ball_detected}}  missing_frames=${{s.ball_missing_frames}}  contact=${{s.contact_flag}}  zone=${{s.pitch_zone}}`];
  const counts = {{}};
  for (const d of s.detections) {{
    const color = COLORS[d.cls_name] || '#fff';
    const x1 = (d.x - d.w / 2) * sx, y1 = (d.y - d.h / 2) * sy;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, d.w * sx, d.h * sy);
    ctx.fillStyle = color; ctx.font = '11px sans-serif';
    ctx.fillText(`${{d.cls_name}} ${{d.confidence.toFixed(2)}}`, x1, Math.max(10, y1 - 3));
    counts[d.cls_name] = (counts[d.cls_name] || 0) + 1;
  }}
  lines.push('counts this frame: ' + JSON.stringify(counts));
  status.textContent = lines.join('\\n');
}}
draw();
</script>
</body></html>
"""


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video", required=True, help="path to the video file (same one run_phase1.py used)"
    )
    ap.add_argument(
        "--clip",
        default=None,
        help="name without extension, e.g. Final. Defaults to --video's filename "
        "stem -- only pass this if your outputs file uses a different stem.",
    )
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    video_stem = video_path.stem
    if args.clip is None:
        args.clip = video_stem
    elif args.clip != video_stem:
        # Hard stop -- a mismatch here means the detections about to be
        # shown were computed from a DIFFERENT video than the one being
        # played, producing a meaningless overlay. No server starts, no
        # browser opens.
        raise SystemExit(
            f"ERROR: --clip '{args.clip}' does not match --video's filename "
            f"('{video_stem}.mp4'). Refusing to serve mismatched detections. "
            f"Either pass --video data/raw/{args.clip}.mp4, or drop --clip "
            f"entirely to auto-derive it from --video."
        )

    scene_path = ROOT / "outputs" / f"{args.clip}_scene_states.json"
    if not scene_path.exists():
        raise SystemExit(f"{scene_path} not found -- run run_phase1.py on this clip first.")

    scenes = json.loads(scene_path.read_text())
    page = PAGE_TEMPLATE.format(
        clip=args.clip,
        colors_json=json.dumps(COLORS),
        **COLORS,
    ).encode("utf-8")
    video_bytes_path = video_path

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # keep the terminal quiet

        def _serve_video(self):
            """
            HTML5 <video> seeking/scrubbing relies on HTTP Range requests --
            the browser asks for just a byte range, not the whole file.
            Advertising Accept-Ranges without honouring Range requests (the
            original version of this method) made the browser repeatedly
            abort and retry on every seek, which is what produced a flood of
            ConnectionAbortedError/ConnectionResetError in the terminal.
            """
            size = video_bytes_path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    _, rng = range_header.split("=")
                    start_s, end_s = rng.split("-")
                    start = int(start_s)
                    end = int(end_s) if end_s else size - 1
                except (ValueError, IndexError):
                    start, end = 0, size - 1
                end = min(end, size - 1)
                length = max(0, end - start + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                start, length = 0, size
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            try:
                with open(video_bytes_path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                # Normal: the browser cancelled this request (e.g. the user
                # seeked again before it finished). Nothing to fix, nothing
                # to report -- just stop sending.
                pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/data":
                body = json.dumps(scenes).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass
            elif self.path == "/video":
                self._serve_video()
            else:
                self.send_response(404)
                self.end_headers()

    port = args.port or free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving tracking viewer for '{args.clip}' at {url}")
    print(f"({len(scenes)} frames, {scenes[0]['timestamp_sec']:.1f}s-{scenes[-1]['timestamp_sec']:.1f}s) -- Ctrl+C to stop")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()