"""Render generated slide sections in headless Chrome (1920x1080) and report elements outside the 128 px margins.

usage: python scripts/slides/check_slide_overflow.py <deck-dir containing project/slides> <deck.css> [slide ids...]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
JS = """
const out = [];
document.querySelectorAll('.wrap').forEach(w => {
  const wr = w.getBoundingClientRect(), id = w.dataset.id, bad = [];
  w.querySelectorAll('*').forEach(el => {
    if (el.closest('aside') || el.tagName==='SECTION') return;
    const r = el.getBoundingClientRect();
    if (!r.width && !r.height) return;
    const right = r.right - wr.left, bottom = r.bottom - wr.top;
    if (right > 1792 + 2 || bottom > 952 + 2 || r.left - wr.left < 126) {
      bad.push(el.tagName + ' ' + Math.round(right) + 'x' + Math.round(bottom) + ' ' + (el.textContent || '').trim().slice(0, 50));
    }
  });
  const s0=w.firstElementChild.getBoundingClientRect(); out.push({id, n: bad.length, bad: bad.slice(0, 4), dbg: Math.round(wr.top)+' '+Math.round(s0.top)+' '+Math.round(s0.height)});
});
document.getElementById('res').textContent = JSON.stringify(out);
"""


def main() -> None:
    deck, css, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    files = sorted((deck / "project" / "slides").glob("*.html"))
    files = [f for f in files if not ids or f.stem in ids]
    body = "".join(f'<div class="wrap" data-id="{f.stem}" style="position:relative;width:1920px;height:1080px;overflow:visible;margin-bottom:40px">{f.read_text()}</div>' for f in files)
    html = (f'<!doctype html><meta charset="utf-8"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@400..700&display=swap" rel="stylesheet">'
            f'<style>{css.read_text()}\n*{{margin:0;box-sizing:border-box}}body{{margin:0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px 14px;text-align:left;border-bottom:2px solid #D5DAE2}}.wrap>section{{width:1920px;height:1080px;box-sizing:border-box;overflow:visible}}aside{{display:none}}</style>'
            f'{body}<pre id="res"></pre><script>window.onload=()=>setTimeout(()=>{{{JS}}},1500)</script>')
    path = Path(tempfile.mkdtemp()) / "check.html"
    path.write_text(html)
    dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=6000", "--window-size=1920,1080",
                          "--dump-dom", f"file://{path}"], capture_output=True, text=True).stdout
    res = dom.split('<pre id="res">')[1].split("</pre>")[0]
    for r in json.loads(res.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")):
        print(f"{r['id']:14s} {'OK' if not r['n'] else str(r['n']) + ' offenders'} [{r['dbg']}]")
        for b in r["bad"]:
            print("      ", b)


if __name__ == "__main__":
    main()
