"""Full-article retrieval for the live news pipeline.

The signal used to judge each story by its headline alone. This module
fetches the article BODY so sentiment sees real context, degrading along
a fallback chain when a site refuses us:

    1. the story representative's own URL
    2. every other outlet in the same story cluster (cluster_headlines
       keeps each copy's URL exactly for this)
    3. a Google News search for the same story on ANY other outlet

Three findings from live testing (2026-07-29) shape the design:
  * ~2/3 of live URLs are news.google.com redirect wrappers that plain
    HTTP cannot follow; they are resolved through Google's own
    batchexecute RPC (the same method the googlenewsdecoder package
    uses), which needs one GET per article plus one shared POST.
  * trafilatura is the cleanest extractor on every domain EXCEPT
    finance.yahoo.com, where it silently truncates at the "Story
    Continues" fold - there the larger of trafilatura and a <p>-cluster
    heuristic wins.
  * Dow Jones properties (WSJ, Barron's, MarketWatch) hard-block plain
    requests with a DataDome 401; investors.com serves a 200 with a
    teaser ("isAccessibleForFree": false). Both mean: walk the chain.

Raw article text is cached under artifacts/cache/fulltext/ which is NOT
whitelisted in .gitignore - copyrighted bodies must never enter the
public repo. Committed artifacts carry only scores and status counters.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, quote_plus, urlparse

import feedparser
import requests

import config

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

_local = threading.local()


def _session() -> requests.Session:
    """One Session per thread (requests.Session is not thread-safe)."""
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _local.session = s
    return _local.session


# --------------------------------------------------------------------------
# Google News URL unwrapping (batchexecute RPC; tested 15/15 on live links)
# --------------------------------------------------------------------------

_BATCH_ENDPOINT = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
_GNEWS_MAP_PATH = config.FULLTEXT_CACHE_DIR / "gnews_map.json"


def is_gnews(url: str | None) -> bool:
    return bool(url) and "news.google.com" in urlparse(url).netloc


def _article_id(gnews_url: str) -> str:
    return urlparse(gnews_url).path.rsplit("/", 1)[-1]


_THROTTLED = "throttled"          # sentinel: back off, don't keep hammering


def _clamped_timeout(deadline: float) -> float:
    return min(config.FULLTEXT_TIMEOUT, max(1.0, deadline - time.monotonic()))


def _get_decoding_params(art_id: str, deadline: float) -> dict | str | None:
    """Scrape the per-article signature + timestamp the RPC requires.
    Returns the _THROTTLED sentinel on 429/5xx so the caller stops
    hammering instead of doubling the request count per article."""
    for base in ("https://news.google.com/rss/articles/",
                 "https://news.google.com/articles/"):
        try:
            r = _session().get(base + art_id,
                               timeout=_clamped_timeout(deadline))
            if r.status_code == 429 or r.status_code >= 500:
                return _THROTTLED
            if r.status_code != 200:
                continue
            sg = re.search(r'data-n-a-sg="([^"]+)"', r.text)
            ts = re.search(r'data-n-a-ts="([^"]+)"', r.text)
            if sg and ts:
                return {"signature": sg.group(1), "timestamp": ts.group(1),
                        "gn_art_id": art_id}
        except requests.RequestException:
            pass
    return None


def _envelope(params: dict, req_id: str) -> list:
    payload = (
        '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,'
        'null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,'
        'null,0],"{gn_art_id}",{timestamp},"{signature}"]'
    ).format(**params)
    return ["Fbv4je", payload, None, req_id]


def _parse_batch_response(text: str) -> dict:
    body = text[4:] if text.startswith(")]}'") else text
    body = body.lstrip("\n")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:            # length-prefixed chunk variant
        data = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("["):
                try:
                    data.extend(json.loads(line))
                except json.JSONDecodeError:
                    pass
    out = {}
    for env in data:
        if (isinstance(env, list) and len(env) >= 3 and env[0] == "wrb.fr"
                and env[1] == "Fbv4je" and isinstance(env[2], str)):
            try:
                out[str(env[-1])] = json.loads(env[2])[1]
            except (json.JSONDecodeError, IndexError, TypeError):
                pass
    return out


def _post_batch(envelopes: list, deadline: float) -> dict:
    payload = f"f.req={quote(json.dumps([envelopes]))}"
    headers = {"content-type":
               "application/x-www-form-urlencoded;charset=UTF-8"}
    r = _session().post(_BATCH_ENDPOINT, headers=headers, data=payload,
                        timeout=_clamped_timeout(deadline))
    r.raise_for_status()
    return _parse_batch_response(r.text)


def _load_json(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _dump_json(path, obj) -> None:
    """Best-effort atomic save. The tmp name is per-process so the 17:00
    task and a dashboard 'regenerate' click can't rename each other's
    half-written file into place; a losing replace() (Windows raises
    PermissionError while the destination is open) is logged, not fatal -
    these are caches."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("cache save failed for %s: %s", path.name, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


_FAIL_RETRY_S = 86400   # failed ids are skipped for a day, then retried


def resolve_gnews_urls(urls: list[str], deadline: float,
                       delay: float = 0.35) -> dict:
    """Resolve news.google.com wrapper URLs -> publisher URLs.

    N politeness-delayed GETs (signature scrape) + one chunked POST per
    25 articles. Article ids are stable: successful resolutions are
    cached on disk permanently, failures for a day (cache values are
    either the URL string or {"failed_ts": ...}). Three consecutive
    throttle responses abort the run's remaining lookups - backing off
    beats hammering a rate limiter. Returns {input_url: url_or_None}.
    """
    cache = _load_json(_GNEWS_MAP_PATH)
    now = time.time()
    out: dict = {}
    envelopes, idmap = [], {}
    throttled = 0
    dirty = False
    for i, u in enumerate(urls):
        aid = _article_id(u)
        hit = cache.get(aid)
        if isinstance(hit, str):
            out[u] = hit
            continue
        if isinstance(hit, dict) and now - hit.get("failed_ts", 0) < _FAIL_RETRY_S:
            out[u] = None
            continue
        if time.monotonic() > deadline or throttled >= 3:
            out[u] = None
            continue
        p = _get_decoding_params(aid, deadline)
        if p == _THROTTLED:
            throttled += 1
            out[u] = None
            time.sleep(delay)
            continue
        throttled = 0
        if p is not None:
            rid = str(i)
            envelopes.append(_envelope(p, rid))
            idmap[rid] = (u, aid)
        else:
            out[u] = None
            cache[aid] = {"failed_ts": now}
            dirty = True
        time.sleep(delay)
    for start in range(0, len(envelopes), 25):
        chunk = envelopes[start:start + 25]
        if time.monotonic() > deadline:
            for env in chunk:
                out[idmap[env[-1]][0]] = None
            continue
        try:
            parsed = _post_batch(chunk, deadline)
        except requests.RequestException as exc:
            log.warning("gnews batch resolve failed: %s", exc)
            parsed = {}
        for env in chunk:
            rid = env[-1]
            u, aid = idmap[rid]
            resolved = parsed.get(rid)
            if isinstance(resolved, str) and resolved.startswith("http"):
                out[u] = cache[aid] = resolved
            else:
                out[u] = None
                cache[aid] = {"failed_ts": now}
            dirty = True
    if dirty:
        _dump_json(_GNEWS_MAP_PATH, cache)
    return out


# --------------------------------------------------------------------------
# Extraction (trafilatura primary; <p>-cluster fallback; yahoo takes longer)
# --------------------------------------------------------------------------

# containers safe to kill by class/id - guarded below so a layout wrapper
# (fool.com's "foolcom-grid-content-sidebar") is never destroyed
_KILL_RE = re.compile(
    r"(related|promo|newsletter|footer|sidebar|share|social|comment|"
    r"breadcrumb|subscription|paywall|advert|cookie|banner|disclaimer|"
    r"recirc)", re.I)
_BAD_P_RE = re.compile(
    r"^(read more|also read|related:|see also|sign up|subscribe|"
    r"advertisement|image source|photo:|getty images|follow \w+ on|"
    r"disclosure:|learn more)", re.I)
# related-article teasers end with an ellipsis (247wallst); legal /
# disclaimer / copyright lines (barrons, ibd, beincrypto)
_JUNK_P_RE = re.compile(
    r"(…|\.\.\.)$|copyright ?©|all rights reserved|"
    r"terms (and conditions|of service)|privacy policy|"
    r"consult with a professional|for any questions or corrections|"
    r"^advertisement|^act now:|email newsroom|comments posted here", re.I)
_ZWS_RE = re.compile("[​‌‍⁠﻿]")   # zero-width (yahoo)
_END_PUNCT_RE = re.compile(r"[.!?:%)\"'”’]$")


def _p_len(node) -> int:
    return sum(len(t) for p in node.find_all("p")
               if len(t := p.get_text(" ", strip=True)) >= 60
               and not _BAD_P_RE.match(t))


def _bs4_extract(html: str) -> str | None:
    """Largest <p>-cluster heuristic: the smallest container that still
    holds ~all of the page's substantial paragraph text."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form",
                     "nav", "header", "footer", "aside", "button",
                     "figcaption"]):
        tag.decompose()
    page_total = _p_len(soup) or 1
    for tag in (soup.find_all(attrs={"class": _KILL_RE})
                + soup.find_all(id=_KILL_RE)):
        try:
            if _p_len(tag) < 0.3 * page_total:  # never kill the article
                tag.decompose()
        except Exception:
            pass
    candidates = soup.find_all(["article", "main", "section", "div"]) or [soup]
    scored = [(n, _p_len(n)) for n in candidates]
    best, best_score = max(scored, key=lambda x: x[1], default=(None, 0))
    if not best_score:
        return None
    for node, sc in sorted(scored, key=lambda x: len(str(x[0]))):
        if sc >= 0.9 * best_score:   # smallest container w/ >=90% of text
            best = node
            break
    paras = [t for p in best.find_all("p")
             if len(t := p.get_text(" ", strip=True)) >= 40
             and not _BAD_P_RE.match(t)]
    return "\n\n".join(paras) or None


def _clean(text: str | None) -> str | None:
    if not text:
        return None
    paras = []
    for p in text.split("\n"):
        q = _ZWS_RE.sub("", p).strip()
        if not q or _JUNK_P_RE.search(q):
            continue
        # short line without sentence punctuation = stray headline /
        # "Most Read" list item (bloomberg-syndicated yahoo pages)
        if len(q) < 90 and not _END_PUNCT_RE.search(q):
            continue
        paras.append(q)
    return "\n\n".join(paras) or None


def extract_text(html: str, url: str | None) -> str | None:
    """Clean article body from fetched HTML, or None."""
    try:
        import trafilatura
        traf = _clean(trafilatura.extract(html, url=url,
                                          include_comments=False,
                                          include_tables=False))
    except Exception:
        traf = None
    try:
        bs4t = _clean(_bs4_extract(html))
    except Exception:
        bs4t = None
    lt, lb = len(traf or ""), len(bs4t or "")
    if url and "finance.yahoo.com" in url:
        # trafilatura truncates yahoo at the "Story Continues" fold
        # (verified 30-65% body loss); take the longer extraction
        text = bs4t if lb > lt else traf
    elif traf and lt >= 300:
        text = traf          # cleaner everywhere else (verified)
    elif lb >= 1000:
        # trafilatura empty but the cluster found a real article; the
        # floor stops subscription pitches from passing as "the article"
        text = bs4t
    else:
        text = traf
    return text or None


# --------------------------------------------------------------------------
# Fetch-outcome classification (verified markers only)
# --------------------------------------------------------------------------

# 429 is deliberately NOT here: throttling is transient and must not be
# cached as a hard block for the TTL window
_BLOCK_STATUSES = {401, 402, 403, 451}
# structural paywall signals (a bare "paywall" substring false-positives
# on open yahoo pages via the enableGPaywallDataStructure JS flag)
_PAYWALL_STRONG = re.compile(
    r'isaccessibleforfree"?\s*:\s*"?false|data-paywall|'
    r'(?:class|id)="[^"]*\bpaywall|87990cbe856818d5eddac44c7b1cdeb8')
_PAYWALL_WEAK = re.compile(
    r"continue reading|subscri(?:be|ption) (?:now )?to (?:continue|read|keep)|"
    r"to read the full (?:story|article)|already a (?:member|subscriber)|"
    r"unlock this article")

_THIN_CHARS = 400        # below this: no usable body
_TEASER_MAX = 1500       # weak markers + shorter than this: paywalled
_FULL_TEXT_MIN = 2500    # strong markers but longer than this: keep it


def fetch_and_extract(url: str) -> dict:
    """GET one publisher URL -> {'status', 'text', 'chars', 'domain'}.

    status: 'ok' (real body), 'partial' (teaser-length text - kept as
    weak context), 'blocked' (hard bot-wall/paywall status), 'empty',
    'error' (transient - retryable next run). Never raises.

    The read is streamed under a byte cap and a TOTAL wall-clock cap:
    requests' timeout only bounds connect and gaps between bytes, so a
    server dripping one byte every few seconds would otherwise hold a
    worker thread - and with it the 17:00 task - indefinitely.
    """
    domain = urlparse(url).netloc
    try:
        r = _session().get(url, timeout=config.FULLTEXT_TIMEOUT,
                           allow_redirects=True, stream=True)
        if r.status_code == 429:
            r.close()
            return {"status": "error", "text": None, "chars": 0,
                    "domain": domain, "url": url, "note": "http 429"}
        if r.status_code in _BLOCK_STATUSES or r.status_code >= 400:
            r.close()
            return {"status": "blocked", "text": None, "chars": 0,
                    "domain": domain, "url": url,
                    "note": f"http {r.status_code}"}
        read_deadline = time.monotonic() + config.FULLTEXT_TIMEOUT
        chunks, size = [], 0
        for chunk in r.iter_content(chunk_size=65536):
            chunks.append(chunk)
            size += len(chunk)
            if size >= config.FULLTEXT_MAX_BYTES:
                break
            if time.monotonic() > read_deadline:
                r.close()
                return {"status": "error", "text": None, "chars": 0,
                        "domain": domain, "url": url, "note": "slow read"}
        enc = r.encoding if (r.encoding
                             and r.encoding.lower() != "iso-8859-1") \
            else "utf-8"
        html = b"".join(chunks).decode(enc, errors="replace")
    except requests.RequestException as exc:
        return {"status": "error", "text": None, "chars": 0,
                "domain": domain, "url": url, "note": type(exc).__name__}
    text = extract_text(html, url) or ""
    low = html.lower()
    paywalled = ((_PAYWALL_STRONG.search(low) and len(text) < _FULL_TEXT_MIN)
                 or (_PAYWALL_WEAK.search(low) and len(text) < _TEASER_MAX))
    if len(text) >= _THIN_CHARS and not paywalled:
        status = "ok"
    elif len(text) >= 150:
        status = "partial"     # teaser/lede: some context beats none
    else:
        return {"status": "empty", "text": None, "chars": 0,
                "domain": domain, "url": url}
    return {"status": status, "text": text, "chars": len(text),
            "domain": domain, "url": url}


# --------------------------------------------------------------------------
# Same-story alternates via the Google News search feed
# --------------------------------------------------------------------------

_ALT_STOP = set("""
a an and are as at be but by can could did do down for from get had has have
her his how in into is it its just me more most my new no not now of off on
or our out over said say says she so some than that the their them then there
these they this those to under up was we were what when where which while who
why will with would you your yours exclusive report reports reported
reportedly breaking update updated live watch video podcast opinion analysis
sources source stock stocks shares
""".split())


def _alt_tokens(title: str) -> set:
    t = re.sub(r"\s+[-|–—]\s+[^-|–—]{2,45}$", "", title)
    return {w.lower() for w in
            re.findall(r"\$?\d[\d,.]*|[A-Za-z][A-Za-z'-]+", t)
            if w.lower() not in _ALT_STOP and len(w) > 1}


def find_alternates(title: str, exclude_sources: set[str],
                    max_results: int = 3) -> list[str]:
    """Google News wrapper URLs for the same story on OTHER outlets,
    best token-overlap first. The feed is relevance-flavored, not
    chronological; when:7d bounds recency."""
    qtoks = _alt_tokens(title)
    if len(qtoks) < 3:
        return []
    query = " ".join(list(qtoks)[:8]) + " when:7d"
    url = ("https://news.google.com/rss/search?q=" + quote_plus(query)
           + "&hl=en-US&gl=US&ceid=US:en")
    try:
        r = _session().get(url, timeout=config.FULLTEXT_TIMEOUT)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
    except Exception as exc:
        log.warning("alternates search failed (%s)", exc)
        return []
    excl = {s.strip().lower() for s in exclude_sources if s}
    hits = []
    for e in feed.entries:
        src = ""
        if hasattr(e, "source"):
            src = (getattr(e, "source", None) or {}).get("title", "")
        if src.strip().lower() in excl:
            continue
        rtoks = _alt_tokens(getattr(e, "title", ""))
        rel = len(qtoks & rtoks) / len(qtoks)
        link = getattr(e, "link", None)
        if rel >= 0.45 and link:
            hits.append((rel, link))
    hits.sort(key=lambda h: -h[0])
    return [link for _, link in hits[:max_results]]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

_BODY_CACHE_PATH = config.FULLTEXT_CACHE_DIR / "bodies.json"


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _fetch_cached(url: str, body_cache: dict, lock: threading.Lock,
                  inflight: dict) -> dict:
    """Fetch-once cache. Concurrent workers asking for the same URL wait
    on the first fetcher's Event instead of duplicating the request (a
    duplicate would waste budget, and a rate-limited duplicate response
    could overwrite a good cached body). Transient 'error' outcomes are
    NOT cached - one bad network moment must not pin a URL as dead for
    the whole TTL window across daily runs."""
    key = _cache_key(url)
    while True:
        with lock:
            hit = body_cache.get(key)
            if hit is not None:
                hit.setdefault("url", url)   # pre-url-era cache entries
                return hit
            ev = inflight.get(key)
            if ev is None:
                ev = inflight[key] = threading.Event()
                break
        ev.wait(timeout=config.FULLTEXT_TIMEOUT * 2 + 5)
        with lock:
            hit = body_cache.get(key)
        if hit is not None:
            hit.setdefault("url", url)
            return hit
        # first fetcher failed transiently and cached nothing
        return {"status": "error", "text": None, "chars": 0,
                "domain": urlparse(url).netloc, "url": url}
    try:
        res = fetch_and_extract(url)
        if res.get("text"):
            res["text"] = res["text"][:config.FULLTEXT_LEAD_CHARS]
            res["chars"] = len(res["text"])   # truth after truncation
        if res["status"] != "error":
            with lock:
                body_cache[key] = {**res, "ts": time.time()}
        return res
    finally:
        with lock:
            inflight.pop(key, None)
        ev.set()


def _attach(story: dict, res: dict, via: str) -> None:
    story["body"] = res["text"]
    story["body_chars"] = res["chars"]
    story["read_status"] = "full" if res["status"] == "ok" else "partial"
    story["read_from"] = res["domain"]
    story["read_url"] = res.get("url")   # the copy actually read
    story["read_via"] = via


def read_stories(stories: list[dict]) -> dict:
    """Fetch article bodies for clustered stories, mutating each story:
    body (lead text), body_chars (chars extracted), read_status
    ('full'|'partial'|'headline_only'), read_from, read_via
    ('direct'|'cluster_fallback'|'alternate').

    Wall-clock bounded by config.FULLTEXT_BUDGET_S; on any failure a
    story simply stays headline_only - the signal must never die here.
    """
    t0 = time.monotonic()
    deadline = t0 + config.FULLTEXT_BUDGET_S
    config.FULLTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    body_cache = _load_json(_BODY_CACHE_PATH)
    cutoff = time.time() - config.FULLTEXT_CACHE_TTL_DAYS * 86400
    body_cache = {k: v for k, v in body_cache.items()
                  if v.get("ts", 0) > cutoff}
    lock = threading.Lock()
    inflight: dict = {}

    for s in stories:
        s.setdefault("read_status", "headline_only")

    # eligible: real news stories with any URL (stocktwits has none)
    todo = [s for s in stories
            if s.get("source") != "stocktwits"
            and any(m.get("url") for m in s.get("members", [{}]))]

    def urls_of(story: dict) -> list[str]:
        seen, out = set(), []
        for m in [story] + story.get("members", []):
            u = m.get("url")
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        return out

    # ---- phase 1: resolve representatives' gnews wrappers (batched) ----
    rep_urls = [s.get("url") for s in todo if s.get("url")]
    resolved = resolve_gnews_urls([u for u in rep_urls if is_gnews(u)],
                                  deadline)

    def final_url(u: str | None) -> str | None:
        if not u:
            return None
        return resolved.get(u, None) if is_gnews(u) else u

    # ---- phase 2: parallel fetch of representative URLs ----
    def _try_direct(story: dict):
        if time.monotonic() > deadline:
            return story, None      # budget spent: bail before fetching
        u = final_url(story.get("url"))
        if not u:
            return story, None
        return story, _fetch_cached(u, body_cache, lock, inflight)

    pending: list[dict] = []
    if time.monotonic() < deadline:
        with ThreadPoolExecutor(max_workers=config.FULLTEXT_WORKERS) as pool:
            futs = [pool.submit(_try_direct, s) for s in todo]
            try:
                for f in as_completed(
                        futs, timeout=max(0.1, deadline - time.monotonic())):
                    try:
                        story, res = f.result()
                    except Exception:
                        continue
                    if res and res["status"] in ("ok", "partial"):
                        _attach(story, res, "direct")
                        if res["status"] == "partial":
                            pending.append(story)  # upgrade via cluster
                    else:
                        pending.append(story)
            except TimeoutError:
                # budget spent: drop queued work; in-flight fetches are
                # bounded by the per-request wall-clock cap
                pool.shutdown(wait=False, cancel_futures=True)
    else:
        pending = list(todo)

    # ---- phase 3: cluster fallback - other outlets' copies ----
    # member URLs are matched by VALUE against the representative's URL:
    # a rep without a URL must not silently shadow its first member
    def _member_urls(story: dict) -> list[str]:
        rep_u = story.get("url")
        return [u for u in urls_of(story) if u != rep_u]

    def _try_members(story: dict):
        best = None
        for u in _member_urls(story):
            if time.monotonic() > deadline:
                break
            u = final_url(u) or (None if is_gnews(u) else u)
            if not u:
                continue
            res = _fetch_cached(u, body_cache, lock, inflight)
            if res["status"] == "ok":
                return story, res
            if res["status"] == "partial" and not best:
                best = res
        return story, best

    if time.monotonic() < deadline and pending:
        member_wrapped = [u for s in pending for u in _member_urls(s)
                          if is_gnews(u) and u not in resolved]
        resolved.update(resolve_gnews_urls(member_wrapped, deadline))
        still: list[dict] = []
        with ThreadPoolExecutor(max_workers=config.FULLTEXT_WORKERS) as pool:
            futs = [pool.submit(_try_members, s) for s in pending]
            try:
                for f in as_completed(
                        futs, timeout=max(0.1, deadline - time.monotonic())):
                    try:
                        story, res = f.result()
                    except Exception:
                        continue
                    if res and res["status"] == "ok":
                        _attach(story, res, "cluster_fallback")
                    elif res and story.get("read_status") == "headline_only":
                        _attach(story, res, "cluster_fallback")
                        still.append(story)
                    elif story.get("read_status") == "headline_only":
                        still.append(story)
            except TimeoutError:
                pool.shutdown(wait=False, cancel_futures=True)
        pending = still

    # ---- phase 4: alternates search for the still-unread, most extreme
    # claimed sentiment first (those sway the advisory most) ----
    n_alt_searches = 0
    if time.monotonic() < deadline:
        pending.sort(key=lambda s: -abs(s.get("score") or 0))
        for story in pending:
            if (n_alt_searches >= config.FULLTEXT_MAX_ALT_SEARCHES
                    or time.monotonic() > deadline):
                break
            if story.get("read_status") == "full":
                continue
            try:
                n_alt_searches += 1
                # exclude outlets already in the cluster; google_news items
                # carry the real publisher as a "... - Publisher" suffix
                excl = {story.get("source", "")} | {
                    m.get("source", "") for m in story.get("members", [])}
                for m in [story] + story.get("members", []):
                    sfx = re.search(r"\s-\s([^-]{2,45})$",
                                    m.get("title") or "")
                    if sfx:
                        excl.add(sfx.group(1).strip())
                alts = find_alternates(story.get("title", ""), excl)
                alt_resolved = resolve_gnews_urls(
                    [u for u in alts if is_gnews(u)], deadline)
                # never re-fetch an article this story already failed on
                tried = {final_url(u) for u in urls_of(story)} - {None}
                for u in alts:
                    pu = alt_resolved.get(u) if is_gnews(u) else u
                    if (not pu or pu in tried
                            or time.monotonic() > deadline):
                        continue
                    res = _fetch_cached(pu, body_cache, lock, inflight)
                    if res["status"] == "ok":
                        _attach(story, res, "alternate")
                        break
            except Exception as exc:
                log.warning("alternate lookup failed for %r: %s",
                            str(story.get("title"))[:60], exc)

    try:
        _dump_json(_BODY_CACHE_PATH, body_cache)
    except OSError as exc:
        log.warning("body cache save failed: %s", exc)

    full = sum(1 for s in stories if s.get("read_status") == "full")
    partial = sum(1 for s in stories if s.get("read_status") == "partial")
    stats = {
        "stories": len(stories),
        "read_full": full,
        "read_partial": partial,
        "headline_only": len(stories) - full - partial,
        "alt_searches": n_alt_searches,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    log.info("fulltext: %(read_full)d full + %(read_partial)d partial of "
             "%(stories)d stories in %(elapsed_s).0fs", stats)
    return stats


# --------------------------------------------------------------------------
# Permanent body corpus (phase 1 of a body-aware brain)
#
# The brain trains on titles because the 710k-article history has no
# bodies. Every body fetched today is training data that cannot be
# recovered later (old pages die), so live-story and top-up bodies are
# appended to a permanent LOCAL parquet. It lives under the gitignored
# cache dir: copyrighted article text must never enter the public repo,
# and it is deliberately excluded from the tracked weekly backups too.
# --------------------------------------------------------------------------

_ARCHIVE_COLS = ["date", "kind", "category", "source", "slug", "title",
                 "url", "status", "chars", "text", "read_via", "fetched_at"]
_ARCHIVE_LOCK = threading.Lock()
_LOCK_STALE_S = 180


def _corpus_lock_path():
    return config.FULLTEXT_ARCHIVE_PATH.with_suffix(".lock")


def _acquire_corpus_lock(timeout_s: float = 30.0) -> int | None:
    """Cross-process mutex via an O_CREAT|O_EXCL sidecar file: the 17:00
    task (topup, then signal) and a dashboard 'Regenerate signal' click
    are SEPARATE processes appending the same parquet - a threading.Lock
    alone was proven to lose whole batches (last writer wins). Locks
    older than _LOCK_STALE_S are broken so a crashed holder can't wedge
    the corpus forever. Returns an fd, or None on timeout."""
    lock = _corpus_lock_path()
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            return fd
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE_S:
                    lock.unlink()
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                return None
            time.sleep(0.25)
        except OSError:
            return None


def _release_corpus_lock(fd: int) -> None:
    try:
        os.close(fd)
        _corpus_lock_path().unlink()
    except OSError:
        pass


def fetch_bodies(urls: list[str], budget_s: float) -> dict:
    """Bulk best-effort body fetch -> {url: result}. Shares the 3-day
    fetch cache with read_stories; same hard budget semantics."""
    deadline = time.monotonic() + budget_s
    config.FULLTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    body_cache = _load_json(_BODY_CACHE_PATH)
    cutoff = time.time() - config.FULLTEXT_CACHE_TTL_DAYS * 86400
    body_cache = {k: v for k, v in body_cache.items()
                  if v.get("ts", 0) > cutoff}
    lock = threading.Lock()
    inflight: dict = {}
    out: dict = {}

    def _one(u: str):
        if time.monotonic() > deadline:
            return u, None
        return u, _fetch_cached(u, body_cache, lock, inflight)

    with ThreadPoolExecutor(max_workers=config.FULLTEXT_WORKERS) as pool:
        futs = [pool.submit(_one, u) for u in urls]
        try:
            for f in as_completed(futs,
                                  timeout=max(0.1, deadline - time.monotonic())):
                try:
                    u, res = f.result()
                except Exception:
                    continue
                if res is not None:
                    out[u] = res
        except TimeoutError:
            pool.shutdown(wait=False, cancel_futures=True)
    _dump_json(_BODY_CACHE_PATH, body_cache)
    return out


def append_bodies(rows: list[dict]) -> int:
    """Append body rows to the permanent corpus; returns net new URLs.

    Dedup is per URL and status-aware: a full body replaces a stored
    partial teaser, never the reverse; ties keep the earliest copy.
    Date semantics: live_story rows carry the ENTRY trading day,
    archive_article rows the GDELT publish day.

    Best-effort for callers (a corpus hiccup never fails topup/signal)
    but never silently destructive: if the cross-process lock or the
    final replace can't be won, the batch spills to a per-pid parquet
    that the next successful append merges back in."""
    import pandas as pd
    rows = [r for r in rows if r.get("url") and r.get("text")]
    if not rows:
        return 0
    new = pd.DataFrame(rows).reindex(columns=_ARCHIVE_COLS)
    new["text"] = new["text"].str.slice(0, config.FULLTEXT_LEAD_CHARS)
    new["chars"] = new["text"].str.len()
    new["fetched_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    new = new[~new["url"].duplicated()]
    path = config.FULLTEXT_ARCHIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    def _spill(df) -> int:
        sp = path.with_suffix(f".spill.{os.getpid()}.parquet")
        try:
            df.to_parquet(sp, index=False)
            log.warning("Body corpus busy: %d rows spilled to %s (merged "
                        "by the next append)", len(df), sp.name)
        except OSError as exc:
            log.warning("Body corpus spill failed too (%s): %d rows lost",
                        exc, len(df))
        return 0

    with _ARCHIVE_LOCK:
        fd = _acquire_corpus_lock()
        if fd is None:
            return _spill(new)
        try:
            frames = [new]
            spills = [p for p in path.parent.glob(
                path.stem + ".spill.*.parquet")]
            for sp in spills:
                try:
                    frames.append(pd.read_parquet(sp)
                                  .reindex(columns=_ARCHIVE_COLS))
                except OSError:
                    pass
            n_old = 0
            if path.exists():
                old = pd.read_parquet(path).reindex(columns=_ARCHIVE_COLS)
                n_old = len(old)
                frames.insert(0, old)
            merged = pd.concat(frames, ignore_index=True)
            merged = (merged
                      .assign(_pref=(merged["status"] != "full").astype(int))
                      .sort_values(["url", "_pref", "chars"],
                                   ascending=[True, True, False],
                                   kind="mergesort")
                      .drop_duplicates("url")
                      .drop(columns="_pref")
                      .sort_values("fetched_at", kind="mergesort")
                      .reset_index(drop=True))
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            for attempt in range(4):
                try:
                    merged.to_parquet(tmp, index=False)
                    tmp.replace(path)
                    break
                except OSError:
                    # a concurrent reader can hold the destination open on
                    # Windows; readers are brief, so short retries win
                    if attempt == 3:
                        try:
                            tmp.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return _spill(new)
                    time.sleep(0.5)
            for sp in spills:
                try:
                    sp.unlink()
                except OSError:
                    pass
            added = max(0, len(merged) - n_old)
            log.info("Body corpus: +%d rows -> %d total", added, len(merged))
            return added
        finally:
            _release_corpus_lock(fd)
