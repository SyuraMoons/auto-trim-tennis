"""Cut and join keep segments with ffmpeg in one pass: frame-accurate, audio kept."""
import subprocess


def build_cmd(src, segments, dst, has_audio=True, crf=20, preset="veryfast"):
    parts, labels = [], []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
        if has_audio:
            parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[a{i}]")
    n = len(segments)
    outs = "[v][a]" if has_audio else "[v]"
    parts.append(f"{''.join(labels)}concat=n={n}:v=1:a={1 if has_audio else 0}{outs}")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", ";".join(parts), "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", dst]
    return cmd


def render(src, segments, dst, has_audio=True, **kw):
    if not segments:
        raise ValueError("no segments to render")
    subprocess.run(build_cmd(src, segments, dst, has_audio, **kw), check=True)
