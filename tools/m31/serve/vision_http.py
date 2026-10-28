"""Bounded OpenAI media ingestion for the opt-in M31 server.

Only this shim installs the video reader extension; shared frontend files stay unchanged.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
import uuid

MARKER = '<__media__>'
VIDEO_PARTS = ('video_url', 'video_file')


def _source(part):
    kind = part.get('type')
    if kind == 'image' and isinstance(part.get('source'), str):
        return part['source'], part.get('media_type', 'image'), part.get('fps')
    if kind in VIDEO_PARTS:
        value = part.get(kind)
        if isinstance(value, dict):
            return value.get('url') or value.get('path') or '', 'video', value.get('fps', part.get('fps'))
        return value or part.get('file') or '', 'video', part.get('fps')
    if kind == 'image':
        source = part.get('source') or {}
        if not isinstance(source, dict):
            raise ValueError('image source must be a string or object')
        if source.get('type') == 'base64':
            return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}", 'image', None
        return source.get('url') or source.get('path') or '', 'image', None
    value = part.get('image_url')
    if isinstance(value, dict):
        value = value.get('url')
    return value or '', 'image', None


def install_video_reader():
    """Extend only this launcher process, before accepting any HTTP request."""
    from serve import frontend
    if getattr(frontend, '_m31_video_reader', False):
        return
    original = frontend._parts_of
    frontend.IMAGE_PARTS += VIDEO_PARTS
    frontend.ANTHROPIC_BLOCKS += VIDEO_PARTS

    def parts(content):
        if not isinstance(content, list) or not any(isinstance(p, dict) and p.get('type') in VIDEO_PARTS for p in content):
            return original(content)
        result = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError('content parts must be objects')
            if part.get('type') in frontend.IMAGE_PARTS:
                source, kind, fps = _source(part)
                result.append({'type': 'image', 'source': source, 'media_type': kind, 'fps': fps})
            elif part.get('type') in frontend.TEXT_PARTS:
                result.append({'type': 'text', 'text': part.get('text', '')})
        return result

    frontend._parts_of = parts
    frontend._m31_video_reader = True


class MediaInput:
    marker = MARKER

    def __init__(self, cache_dir, *, max_media_bytes=32*1024*1024,
                 max_total_bytes=128*1024*1024, max_media=32,
                 max_image_pixels=16_000_000, max_video_seconds=120,
                 max_video_frames=120, max_video_fps=30, video_fps=1.0,
                 allow_http=True, allow_local=True):
        self.dir = Path(cache_dir).resolve() / ('sources-' + uuid.uuid4().hex)
        self.dir.mkdir(parents=True, exist_ok=True)
        for value in (max_media_bytes, max_total_bytes, max_media, max_image_pixels, max_video_frames):
            if type(value) is not int or value <= 0:
                raise ValueError('vision size/count limits must be positive integers')
        for value in (max_video_seconds, max_video_fps):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('vision time/fps limits must be finite and positive')
        if not isinstance(allow_http, bool) or not isinstance(allow_local, bool):
            raise ValueError('vision allow_http/allow_local must be boolean')
        self.max_media_bytes, self.max_total_bytes = max_media_bytes, max_total_bytes
        self.max_media = max_media
        self.max_image_pixels, self.max_video_seconds = max_image_pixels, float(max_video_seconds)
        self.max_video_frames, self.max_video_fps = max_video_frames, float(max_video_fps)
        self.video_fps = video_fps
        self.allow_http, self.allow_local = allow_http, allow_local
        self.lock = threading.Lock()
        self.references = {}
        self.last_timings = {}

    def _fps(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= self.max_video_fps:
            raise ValueError(f'video_fps must be finite and in (0, {self.max_video_fps}]')
        return float(value)

    def _read(self, source):
        if not isinstance(source, str) or not source:
            raise ValueError('media URL/path must be a nonempty string')
        scheme = urlparse(source).scheme
        if scheme == 'data':
            try:
                header, payload = source.split(',', 1)
                if ';base64' not in header or not header.startswith(('data:image/', 'data:video/')):
                    raise ValueError('media data URL needs image/video MIME and base64 encoding')
                if len(payload) > 4*((self.max_media_bytes+2)//3):
                    raise ValueError('media exceeds max_media_bytes')
                started = time.perf_counter_ns()
                data = base64.b64decode(payload, validate=True)
                self.last_timings['base64_decode_ms'] = self.last_timings.get('base64_decode_ms', 0) + (time.perf_counter_ns()-started)/1e6
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f'invalid media data URL: {exc}') from exc
        elif scheme in ('http', 'https'):
            if not self.allow_http:
                raise ValueError('HTTP media URLs are disabled; send a base64 data URL or local file')
            with urlopen(source, timeout=15) as response:
                data = response.read(self.max_media_bytes+1)
        elif scheme in ('', 'file'):
            if not self.allow_local:
                raise ValueError('local media files are disabled')
            if scheme == 'file' and urlparse(source).netloc not in ('', 'localhost'):
                raise ValueError('file URL must be local')
            path = Path(unquote(urlparse(source).path) if scheme else source)
            if not path.is_file() or path.stat().st_size > self.max_media_bytes:
                raise ValueError('media is not a regular file or exceeds max_media_bytes')
            with path.open('rb') as stream:
                data = stream.read(self.max_media_bytes+1)
        else:
            raise ValueError('unsupported media URL scheme')
        if not data or len(data) > self.max_media_bytes:
            raise ValueError('empty media or media exceeds max_media_bytes')
        return data

    def _store(self, data, kind, fps):
        stored_started = time.perf_counter_ns()
        path = self.dir / (hashlib.sha256(data).hexdigest() + ('.video' if kind == 'video' else '.image'))
        with self.lock:
            if not path.exists():
                temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
                try:
                    temp.write_bytes(data)
                    os.replace(temp, path)
                finally:
                    if temp.exists():
                        temp.unlink()
            self.references[path] = self.references.get(path, 0) + 1
        self.last_timings['source_hash_store_ms'] = self.last_timings.get('source_hash_store_ms', 0) + (time.perf_counter_ns()-stored_started)/1e6
        try:
            self._validate(path, kind, fps)
        except BaseException:
            self.release([path])
            raise
        return path

    def _validate(self, path, kind, fps):
        validated_started = time.perf_counter_ns()
        if kind == 'image':
            from PIL import Image
            try:
                with Image.open(path) as image:
                    if image.width * image.height > self.max_image_pixels:
                        raise ValueError('image exceeds max_image_pixels')
                    image.verify()
            except (OSError, Image.DecompressionBombError) as exc:
                raise ValueError(f'invalid image: {exc}') from exc
        else:
            try:
                probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                        '-show_entries', 'stream=width,height:format=duration', '-of', 'json', str(path)],
                                       capture_output=True, text=True, timeout=15, check=True)
                info = json.loads(probe.stdout)
                duration = float(info['format']['duration'])
                stream = info['streams'][0]
                if not math.isfinite(duration) or duration <= 0 or duration > self.max_video_seconds:
                    raise ValueError('video exceeds max_video_seconds or has invalid duration')
                if math.ceil(duration*fps) > self.max_video_frames:
                    raise ValueError('video exceeds max_video_frames at selected fps')
                if int(stream['width'])*int(stream['height']) > self.max_image_pixels:
                    raise ValueError('video dimensions exceed max_image_pixels')
            except (subprocess.SubprocessError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f'invalid video: {exc}') from exc
        key = 'pil_verify_ms' if kind == 'image' else 'ffprobe_ms'
        self.last_timings[key] = self.last_timings.get(key, 0) + (time.perf_counter_ns()-validated_started)/1e6

    def release(self, paths):
        """Prepared requests keep their own source references until encoding ends."""
        with self.lock:
            for path in paths:
                count = self.references.get(path, 0)
                if count <= 0:
                    continue
                if count == 1:
                    self.references.pop(path)
                    path.unlink(missing_ok=True)
                else:
                    self.references[path] = count - 1

    def close(self):
        # This random per-instance directory contains only this reader's files.
        shutil.rmtree(self.dir, ignore_errors=True)

    def prepare(self, messages, requested_fps=None, cancel=None):
        """Return text-only helper messages, ordered immutable media files and marker substitutions."""
        self.last_timings = {}
        copied = copy.deepcopy(messages)
        media, markers, total, selected_fps = [], [], 0, None
        default_fps = self._fps(self.video_fps if requested_fps is None else requested_fps)
        try:
            for message in copied:
                if cancel is not None and cancel.is_set():
                    from serve.server import ClientGone
                    raise ClientGone()
                if not isinstance(message.get('content'), list):
                    continue
                parts = []
                for part in message['content']:
                    if cancel is not None and cancel.is_set():
                        from serve.server import ClientGone
                        raise ClientGone()
                    if not isinstance(part, dict):
                        raise ValueError('content parts must be objects')
                    kind = part.get('type')
                    if kind in ('text', 'input_text', None):
                        if not isinstance(part.get('text'), str):
                            raise ValueError('text content must be a string')
                        parts.append(part['text'])
                        continue
                    if kind not in ('image', 'image_url', 'input_image', *VIDEO_PARTS):
                        raise ValueError(f'unsupported M31 content part: {kind}')
                    if len(media) >= self.max_media:
                        raise ValueError('request exceeds max_media')
                    source, media_kind, fps = _source(part)
                    fps = self._fps(default_fps if fps is None else fps)
                    if media_kind == 'video':
                        if selected_fps is not None and fps != selected_fps:
                            raise ValueError('all videos in one request must use the same fps')
                        selected_fps = fps
                    read_started = time.perf_counter_ns()
                    data = self._read(source)
                    self.last_timings['source_read_including_base64_ms'] = self.last_timings.get('source_read_including_base64_ms', 0) + (time.perf_counter_ns()-read_started)/1e6
                    total += len(data)
                    if total > self.max_total_bytes:
                        raise ValueError('request exceeds max_total_bytes')
                    media.append(self._store(data, media_kind, fps))
                    marker = 'M31_MEDIA_' + uuid.uuid4().hex
                    markers.append(marker)
                    parts.append(marker)
                message['content'] = ''.join(parts)
            # An unescaped mtmd marker in user text would consume a different media item.
            if media and any(MARKER in str(m.get('content', '')) for m in copied):
                raise ValueError('literal mtmd media marker is reserved in multimodal requests')
        except BaseException:
            self.release(media)
            raise
        return (copied if media else copy.deepcopy(messages)), media, markers, selected_fps or default_fps
