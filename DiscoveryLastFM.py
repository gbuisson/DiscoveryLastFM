#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiscoveryLastFM.py · v2.1.1
– Lidarr/Headphones integration via modular service layer
– Support for both services with configuration switch
– GitHub auto-update system with backup and rollback
– Maintains identical workflow and cache compatibility with v1.7.x
– Zero breaking changes for existing configurations
"""

# ─────────────────── CONFIG ───────────────────
# Try to import configuration from config.py, fallback to defaults
try:
    from config import *
except ImportError:
    print("Warning: config.py not found. Using example values.")
    print("Please copy config.example.py to config.py and update with your credentials.")
    # Default/example values - these should be overridden in config.py
    LASTFM_USERNAME = "your_lastfm_username"
    LASTFM_API_KEY = "your_lastfm_api_key"
    HP_API_KEY = "your_headphones_api_key"
    HP_ENDPOINT = "http://your-headphones-server:port"
    MUSIC_SERVICE = "headphones"

# Default configuration values (can be overridden in config.py)
if 'RECENT_MONTHS' not in globals():
    RECENT_MONTHS = 3
if 'MIN_PLAYS' not in globals():
    MIN_PLAYS = 20
if 'REQUEST_LIMIT' not in globals():
    REQUEST_LIMIT = 1/5
if 'MBZ_DELAY' not in globals():
    MBZ_DELAY = 1.1
if 'SIMILAR_MATCH_MIN' not in globals():
    SIMILAR_MATCH_MIN = 0.46
if 'MAX_SIMILAR_PER_ART' not in globals():
    MAX_SIMILAR_PER_ART = 20
if 'MAX_POP_ALBUMS' not in globals():
    MAX_POP_ALBUMS = 5
if 'CACHE_TTL_HOURS' not in globals():
    CACHE_TTL_HOURS = 24
if 'DEBUG_PRINT' not in globals():
    DEBUG_PRINT = True
if 'MUSIC_SERVICE' not in globals():
    MUSIC_SERVICE = "headphones"

BAD_SEC = {
    "Compilation", "Live", "Remix", "Soundtrack", "DJ-Mix",
    "Mixtape/Street", "EP", "Single", "Interview", "Audiobook"
}

# ─────────────────── LIBRARIES ───────────────────
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import json, logging, os, sys, time, urllib.parse, requests

# Import new service layer
from services import MusicServiceFactory, ArtistInfo, AlbumInfo, ServiceError, ConfigurationError

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_FILE = SCRIPT_DIR / "lastfm_similar_cache.json"
LOG_DIR = SCRIPT_DIR / "log"
LOG_FILE = LOG_DIR / "discover.log"

# ─────────────────── LOGGER ───────────────────
# Ensure the log directory exists
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ],
)
log = logging.getLogger("lfm2hp")

# Reduce verbosity of requests loggers to avoid duplication
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)

def dprint(msg):
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")

# ──────────── RATE LIMIT WRAPPERS ────────────
def rate_limited(delay):
    def decorator(fn):
        last = 0
        def wrapped(*args, **kwargs):
            nonlocal last
            wait = delay - (time.time() - last)
            if wait > 0:
                dprint(f"sleep {wait:.2f}s ({fn.__name__})")
                time.sleep(wait)
            result = fn(*args, **kwargs)
            last = time.time()
            return result
        return wrapped
    return decorator

@rate_limited(REQUEST_LIMIT)
def lf_request(method, **params):
    # Last.fm API call with robust retry handling
    for alt, real in (("from_", "from"), ("to_", "to")):
        if alt in params:
            params[real] = params.pop(alt)

    base = "https://ws.audioscrobbler.com/2.0/"
    params |= {"method": method, "api_key": LASTFM_API_KEY, "format": "json"}

    # Retry configuration
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            dprint(f"LF  → {base}?{urllib.parse.urlencode(params)} (attempt {attempt+1}/{max_retries})")
            r = requests.get(base, params=params, timeout=15)
            dprint(f"LF  ← {r.status_code}")

            # Rate limiting
            if r.status_code == 429 and attempt < max_retries - 1:
                wait_time = int(r.headers.get('Retry-After', retry_delay * 2))
                log.warning(f"Rate limit Last.fm, waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            if r.status_code != 200:
                if attempt < max_retries - 1:
                    log.warning(f"Last.fm HTTP {r.status_code}, attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    log.warning(f"Last.fm HTTP {r.status_code}: {r.text[:200]}")
                    return None

            try:
                return r.json()
            except:
                if attempt < max_retries - 1:
                    log.warning(f"Last.fm invalid JSON, attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                else:
                    log.warning(f"Last.fm invalid JSON: {r.text[:200]}")
                    return None

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                OSError) as e:
            if attempt < max_retries - 1:
                log.warning(f"Last.fm connection error: {e}, attempt {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                log.warning(f"Last.fm connection failed after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            # Catch any wrapped connection errors (e.g., from urllib3)
            error_str = str(e).lower()
            if any(err in error_str for err in ['connection', 'reset', 'aborted', 'timeout', 'broken pipe']):
                if attempt < max_retries - 1:
                    log.warning(f"Last.fm network error: {e}, attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    log.warning(f"Last.fm network failed after {max_retries} attempts: {e}")
                    return None
            if attempt < max_retries - 1:
                log.warning(f"Last.fm error: {e}, attempt {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                log.warning(f"Last.fm error after {max_retries} attempts: {e}")
                return None

    return None

@rate_limited(MBZ_DELAY)
def mbz_request(path, **params):
    # MusicBrainz API call with robust retry handling
    base = "https://musicbrainz.org/ws/2/"
    params.setdefault("fmt", "json")

    headers = {"User-Agent": "DiscoveryLastFM/2.0.0 ( mrroboto@example.com )"}

    # Retry configuration for MusicBrainz
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            dprint(f"MBZ → {base}{path}?{urllib.parse.urlencode(params)} (attempt {attempt+1}/{max_retries})")
            r = requests.get(
                base + path, params=params, headers=headers, timeout=30
            )
            dprint(f"MBZ ← {r.status_code}")

            # Handle MusicBrainz rate limiting (code 429)
            if r.status_code == 429 and attempt < max_retries - 1:
                wait_time = int(r.headers.get('Retry-After', retry_delay * 2))
                log.warning(f"Rate limit MusicBrainz, waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            if r.status_code != 200:
                if attempt < max_retries - 1:
                    log.warning(f"MusicBrainz HTTP {r.status_code}, attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    log.warning(f"MusicBrainz HTTP {r.status_code}: {r.text[:200]}")
                    return None

            try:
                return r.json()
            except:
                if attempt < max_retries - 1:
                    log.warning(f"MusicBrainz invalid JSON, attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                else:
                    log.warning(f"MusicBrainz invalid JSON: {r.text[:200]}")
                    return None

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                OSError) as e:
            if attempt < max_retries - 1:
                log.warning(f"MusicBrainz connection error: {e}, attempt {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                log.warning(f"MusicBrainz connection failed after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            # Catch any wrapped connection errors (e.g., from urllib3)
            error_str = str(e).lower()
            if any(err in error_str for err in ['connection', 'reset', 'aborted', 'timeout', 'broken pipe']):
                if attempt < max_retries - 1:
                    log.warning(f"MusicBrainz network error: {e}, attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    log.warning(f"MusicBrainz network failed after {max_retries} attempts: {e}")
                    return None
            if attempt < max_retries - 1:
                log.warning(f"MusicBrainz error: {e}, attempt {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                log.warning(f"MusicBrainz error after {max_retries} attempts: {e}")
                return None

    return None

# ────────────── CORE FUNCTIONS (UNCHANGED) ──────────────
def load_cache():
    """Load cache from JSON file"""
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)

        # Ensure added_albums is a set
        if "added_albums" in cache and isinstance(cache["added_albums"], list):
            cache["added_albums"] = set(cache["added_albums"])
        elif "added_albums" not in cache:
            cache["added_albums"] = set()

        return cache
    except:
        return {"similar_cache": {}, "added_albums": set()}

def save_cache(cache):
    """Save cache to JSON file with memory-efficient handling"""
    try:
        # Convert set to list for JSON without duplicating memory
        cache_to_save = {}
        for key, value in cache.items():
            if key == "added_albums" and isinstance(value, set):
                cache_to_save[key] = list(value)
            else:
                cache_to_save[key] = value

        # Save to temp file then rename for atomicity
        temp_file = CACHE_FILE.with_suffix('.tmp')
        with open(temp_file, "w") as f:
            json.dump(cache_to_save, f, indent=2, separators=(',', ':'))

        # Rename atomically
        temp_file.replace(CACHE_FILE)

    except Exception as e:
        log.error(f"Cache save error: {e}")
        # Cleanup temp file if it exists
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()

def recent_artists():
    """Get recently listened artists with optimized memory handling"""
    end = int(time.time())
    start = end - (RECENT_MONTHS * 30 * 24 * 3600)

    # Handle pagination for large datasets
    artist_plays = defaultdict(int)
    page = 1
    total_pages = 1
    processed_tracks = 0

    while page <= total_pages and page <= 10:  # Max 10 pages for safety
        js = lf_request("user.getRecentTracks", user=LASTFM_USERNAME, from_=start, to_=end, limit=200, page=page)
        if not js:
            break

        recenttracks = js.get("recenttracks", {})
        tracks = recenttracks.get("track", [])

        # Update total_pages from first response
        if page == 1:
            attr = recenttracks.get("@attr", {})
            total_pages = min(int(attr.get("totalPages", 1)), 10)  # Page limit
            log.info(f"Processing {attr.get('total', 0)} recent tracks across {total_pages} pages")

        # Process tracks from current page
        for t in tracks:
            if isinstance(t, dict):
                artist = t.get("artist", {})
                name = artist.get("#text", "") if isinstance(artist, dict) else str(artist)
                if name:
                    artist_plays[name] += 1
                    processed_tracks += 1

        page += 1

    log.info(f"Processed {processed_tracks} tracks from {len(artist_plays)} unique artists")

    # Filter and get MBID for artists with enough plays
    result = []
    qualifying_artists = [(name, plays) for name, plays in artist_plays.items() if plays >= MIN_PLAYS]
    log.info(f"Found {len(qualifying_artists)} artists with ≥{MIN_PLAYS} plays")

    for name, plays in qualifying_artists:
        js = lf_request("artist.getInfo", artist=name)
        if js:
            mbid = js.get("artist", {}).get("mbid")
            if mbid:
                result.append((name, mbid))
                log.debug(f"Artist {name}: {plays} plays, MBID: {mbid}")

    log.info(f"Final result: {len(result)} artists with valid MBIDs")
    return result

def cached_similars(cache, aid):
    """Check similar artists cache with TTL - UNCHANGED"""
    if aid not in cache["similar_cache"]:
        return None

    entry = cache["similar_cache"][aid]
    age_hours = (time.time() - entry["ts"]) / 3600

    if age_hours > CACHE_TTL_HOURS:
        dprint(f"Cache expired for {aid} ({age_hours:.1f}h)")
        return None

    return entry["data"]

def top_albums(artist_mbid):
    """Get filtered popular albums - UNCHANGED"""
    js = lf_request("artist.getTopAlbums", mbid=artist_mbid, limit=MAX_POP_ALBUMS*2)
    if not js:
        return []

    albums = js.get("topalbums", {}).get("album", [])
    return [a.get("mbid") for a in albums if a.get("mbid")][:MAX_POP_ALBUMS]

def release_to_rg(rel_id):
    """Convert Release ID to Release Group ID - UNCHANGED"""
    if not rel_id:
        return None

    js = mbz_request(f"release/{rel_id}", inc="release-groups")
    if js and "release-group" in js:
        return js["release-group"]["id"]
    return None

def is_studio_rg(rg_id):
    """Check if it's a studio album - UNCHANGED"""
    if not rg_id:
        return None

    js = mbz_request(f"release-group/{rg_id}")
    if not js:
        return None

    # Check primary type
    primary = js.get("primary-type")
    if primary != "Album":
        return False

    # Check secondary types
    secondary = js.get("secondary-types", [])
    if any(s in BAD_SEC for s in secondary):
        return False

    return True

# ────────────── MUSIC SERVICE INTEGRATION ──────────────
def validate_configuration():
    """Extended validation for all services with detailed checks"""
    config_dict = {k: v for k, v in globals().items() if k.isupper()}
    service_type = config_dict.get("MUSIC_SERVICE", "headphones").lower()

    log.info("Starting configuration validation...")

    # Base parameters validation
    required_base = ["LASTFM_USERNAME", "LASTFM_API_KEY"]
    missing_base = [k for k in required_base if not config_dict.get(k)]
    if missing_base:
        raise ConfigurationError(f"Missing base configuration: {missing_base}")

    # Numeric parameters validation
    numeric_params = {
        "RECENT_MONTHS": (1, 12),
        "MIN_PLAYS": (1, 1000),
        "MAX_SIMILAR_PER_ART": (1, 100),
        "MAX_POP_ALBUMS": (1, 50),
        "CACHE_TTL_HOURS": (1, 168)  # 1 week max
    }

    for param, (min_val, max_val) in numeric_params.items():
        value = config_dict.get(param)
        if value is not None:
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                raise ConfigurationError(f"{param} must be between {min_val} and {max_val}, got {value}")

    # Rate limiting validation
    request_limit = config_dict.get("REQUEST_LIMIT", 1/5)
    if request_limit <= 0 or request_limit > 10:
        raise ConfigurationError(f"REQUEST_LIMIT must be between 0 and 10, got {request_limit}")

    mbz_delay = config_dict.get("MBZ_DELAY", 1.1)
    if mbz_delay < 0.5 or mbz_delay > 10:
        raise ConfigurationError(f"MBZ_DELAY must be between 0.5 and 10, got {mbz_delay}")

    # Service-specific validation
    if not MusicServiceFactory.validate_service_config(service_type, config_dict):
        available = ", ".join(MusicServiceFactory.get_available_services())
        raise ConfigurationError(
            f"Invalid configuration for {service_type}. "
            f"Available services: {available}"
        )

    log.info(f"Configuration validated successfully for {service_type}")
    log.info(f"- Discovery scope: {config_dict.get('RECENT_MONTHS', 3)} months, {config_dict.get('MIN_PLAYS', 20)} min plays")
    log.info(f"- Rate limits: LastFM {request_limit}/s, MusicBrainz {mbz_delay}s delay")
    log.info(f"- Processing limits: {config_dict.get('MAX_SIMILAR_PER_ART', 20)} similar artists, {config_dict.get('MAX_POP_ALBUMS', 5)} albums each")

def sync():
    """Sync function modified for service abstraction"""
    start_time = time.time()

    try:
        # Service initialization
        config_dict = {k: v for k, v in globals().items() if k.isupper()}
        service_type = config_dict.get("MUSIC_SERVICE", "headphones")

        music_service = MusicServiceFactory.create_service(service_type, config_dict)
        log.info(f"Using {service_type} service: {music_service.get_service_info()}")

        # Rest of workflow UNCHANGED from v1.7.x
        cache = load_cache()
        added_albums = set(cache.get("added_albums", []))
        recent = recent_artists()

        log.info("Analyzing %d artists...", len(recent))

        seen = set()
        fallback_ids = []
        success_count = 0
        error_count = 0
        skipped_count = 0

        for name, aid in recent:
            if not aid:
                continue

            log.info(f"Processing artist: {name} ({aid})")

            # Convert to structured data
            artist_info = ArtistInfo(mbid=aid, name=name)

            # Add artist (SAME LOGIC, different implementation)
            try:
                if not music_service.add_artist(artist_info):
                    log.error(f"Unable to add artist {name} ({aid})")
                    error_count += 1
                    continue
            except ServiceError as e:
                log.error(f"Service error adding artist {name}: {e}")
                error_count += 1
                continue
            except Exception as e:
                log.error(f"Unexpected error adding artist {name}: {e}")
                error_count += 1
                continue

            # Refresh artist
            music_service.refresh_artist(aid)

            # Handle similar artists - UNCHANGED WORKFLOW (with optimized cache)
            sims = cached_similars(cache, aid)
            if not sims:
                log.info(f"Searching similar artists for {name}...")
                js = lf_request("artist.getSimilar", mbid=aid, limit=50)
                sims = js.get("similarartists", {}).get("artist", []) if js else []
                # Save in cache only if we have valid data
                if sims:
                    cache["similar_cache"][aid] = {"ts": time.time(), "data": sims}

            proc = 0
            for s in sims:
                sim_name = s.get("name", "Unknown")
                sid = s.get("mbid")
                sim_match = float(s.get("match", 0))

                if proc >= MAX_SIMILAR_PER_ART:
                    log.debug(f"Skipping {sim_name} ({sid}): exceeded MAX_SIMILAR_PER_ART")
                    break
                if not sid:
                    log.debug(f"Skipping {sim_name}: missing MBID")
                    continue
                if sid in seen:
                    log.debug(f"Skipping {sim_name} ({sid}): already processed")
                    continue
                if sim_match < SIMILAR_MATCH_MIN:
                    log.debug(f"Skipping {sim_name} ({sid}): match too low ({sim_match})")
                    continue

                seen.add(sid)
                proc += 1
                log.info(f"Processing similar artist: {sim_name} ({sid})")

                # Add similar artist with service layer
                similar_artist_info = ArtistInfo(mbid=sid, name=sim_name)
                try:
                    if not music_service.add_artist(similar_artist_info):
                        log.error(f"Unable to add similar artist {sim_name} ({sid})")
                        error_count += 1
                        continue
                except ServiceError as e:
                    log.error(f"Service error adding similar artist {sim_name}: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    log.error(f"Unexpected error adding similar artist {sim_name}: {e}")
                    error_count += 1
                    continue

                music_service.refresh_artist(sid)

                # Process similar artist's albums - UNCHANGED LOGIC
                albums = top_albums(sid)
                log.info(f"Found {len(albums)} albums for {sim_name}")

                # Also retrieve the original list with titles
                js_albums = lf_request("artist.getTopAlbums", mbid=sid, limit=MAX_POP_ALBUMS*2)
                albums_raw = js_albums.get("topalbums", {}).get("album", []) if js_albums else []
                mbid_to_title = {a.get("mbid"): a.get("name") for a in albums_raw if a.get("mbid")}

                for rel_id in albums:
                    rg_id = release_to_rg(rel_id)
                    title = mbid_to_title.get(rel_id, rel_id)

                    if not rg_id:
                        # Fallback: missing MBID, use artist name and album title
                        log.info(f"Fallback: adding album without MBID (artist: {sim_name}, title: {title})")
                        # Skip fallback in service layer for now - future implementation
                        continue

                    # Check album existence using service layer
                    if music_service.album_exists(rg_id, added_albums) or music_service.album_exists(rel_id, added_albums):
                        log.debug(f"Album {rel_id} already exists")
                        skipped_count += 1
                        continue

                    studio = is_studio_rg(rg_id)
                    if studio is False:
                        log.debug(f"Album {rel_id} is not a studio album")
                        continue

                    try:
                        # Convert to AlbumInfo for service layer
                        album_info = AlbumInfo(
                            mbid=rg_id if studio else rel_id,
                            title=title,
                            artist_mbid=sid,
                            artist_name=sim_name
                        )

                        # If we don't know if it's studio, use the release ID
                        if studio is None:
                            log.info(f"Adding album (fallback) {rel_id}")
                            album_info.mbid = rel_id
                            fallback_ids.append(rel_id)
                        else:
                            log.info(f"Adding album {rg_id}")

                        # Add album with service layer
                        try:
                            if music_service.add_album(album_info):
                                # Queue album
                                if music_service.queue_album(album_info, force_new=True):
                                    added_albums.add(album_info.mbid)
                                    success_count += 1

                                    # Note: cache saved in batch at the end for performance
                                else:
                                    log.warning(f"Album {album_info.title} added but queue failed")
                            else:
                                error_count += 1
                                log.error(f"Failed to add album {album_info.title}")
                        except ServiceError as e:
                            error_count += 1
                            log.error(f"Service error adding album {album_info.title}: {e}")
                        except Exception as e:
                            error_count += 1
                            log.error(f"Unexpected error adding album {album_info.title}: {e}")

                    except Exception as e:
                        error_count += 1
                        log.error(f"Error while adding album {rel_id or rg_id}: {e}")

        # Final force search
        if fallback_ids:
            log.info(f"Final update for {len(fallback_ids)} albums...")
            try:
                music_service.force_search()
            except ServiceError as e:
                log.error(f"Force search failed: {e}")

        # Final cache save for performance
        cache["added_albums"] = list(added_albums)
        save_cache(cache)

        # UNCHANGED statistics
        elapsed_time = time.time() - start_time
        log.info("Sync completed in %.1f minutes.", elapsed_time / 60)
        log.info("- Albums added: %d", success_count)
        log.info("- Errors: %d", error_count)
        log.info("- Skipped: %d", skipped_count)
        log.info("- Fallback: %d", len(fallback_ids))

    except (ServiceError, ConfigurationError) as e:
        log.error(f"Service error: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        raise
    finally:
        # Final cache save
        cache["added_albums"] = list(added_albums)
        save_cache(cache)

        # Memory cleanup
        import gc
        gc.collect()
        log.debug("Memory cleanup completed")

def handle_update_command():
    """Handle the --update command"""
    from utils.updater import create_updater_from_config, get_current_version

    # Updater configuration
    config_dict = {k: v for k, v in globals().items() if k.isupper()}
    config_dict["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

    updater = create_updater_from_config(config_dict)

    print(f"DiscoveryLastFM Auto-Update System")
    print(f"Current version: {get_current_version()}")
    print(f"Repository: {updater.repo_owner}/{updater.repo_name}")
    print()

    # Check for updates
    print("Checking for updates...")
    release_info = updater.check_for_updates()

    if not release_info:
        print("✅ Already up to date!")
        return

    # Show new version info
    print(f"🆕 Update available: {release_info['version']}")
    print(f"   Release: {release_info['name']}")
    print(f"   Published: {release_info['published_at']}")

    if release_info.get('prerelease'):
        print("   ⚠️  This is a pre-release version")

    print()
    print("Release Notes:")
    print("-" * 50)
    print(release_info['body'][:500] + ("..." if len(release_info['body']) > 500 else ""))
    print("-" * 50)
    print()

    # Confirm update
    while True:
        response = input("Do you want to install this update? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no', '']:
            print("Update cancelled.")
            return
        else:
            print("Please enter 'y' or 'n'")

    # Execute update
    print("\n🚀 Starting update process...")
    print("This will:")
    print("1. Create a backup of current version")
    print("2. Download and install the new version")
    print("3. Verify the installation")
    print("4. Rollback automatically if anything goes wrong")
    print()

    success = updater.perform_update(release_info)

    if success:
        print("✅ Update completed successfully!")
        print(f"   Updated to version: {release_info['version']}")
        print("   Your configuration and cache files have been preserved.")
        print("\n🔄 Please restart the application to use the new version.")
    else:
        print("❌ Update failed!")
        print("   Your previous version has been restored.")
        print("   Check the logs for more details.")


def handle_update_status():
    """Show the updater status"""
    from utils.updater import create_updater_from_config

    config_dict = {k: v for k, v in globals().items() if k.isupper()}
    config_dict["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

    updater = create_updater_from_config(config_dict)
    status = updater.get_update_status()

    print("DiscoveryLastFM Update Status")
    print("=" * 40)
    print(f"Current Version: {status['current_version']}")
    print(f"Repository: {status['repo']}")
    print(f"Auto-update: {'Enabled' if status['auto_update_enabled'] else 'Disabled'}")

    if status['last_check']:
        print(f"Last Check: {status['last_check']}")
    else:
        print("Last Check: Never")

    if status['available_version']:
        if status['available_version'] != status['current_version']:
            print(f"Available Version: {status['available_version']} ⚠️")
        else:
            print(f"Available Version: {status['available_version']} ✅")

    if status['failed_attempts'] > 0:
        print(f"Failed Attempts: {status['failed_attempts']} ❌")

    print(f"Backups Available: {status['backup_count']}")

    if status['next_check']:
        print(f"Next Check: {status['next_check']}")


def handle_backups_list():
    """List available backups"""
    from utils.updater import create_updater_from_config

    config_dict = {k: v for k, v in globals().items() if k.isupper()}
    config_dict["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

    updater = create_updater_from_config(config_dict)
    backups = updater.list_backups()

    if not backups:
        print("No backups found.")
        return

    print("Available Backups")
    print("=" * 60)
    print(f"{'Version':<10} {'Date':<20} {'Size':<10} {'Status'}")
    print("-" * 60)

    for backup in backups:
        status = "✅ OK" if backup['exists'] else "❌ Missing"
        timestamp = backup['timestamp']
        # Format timestamp
        try:
            from datetime import datetime
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = timestamp

        print(f"{backup['version']:<10} {date_str:<20} {backup['size_mb']} MB{'':<5} {status}")


def parse_cli_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description='DiscoveryLastFM - Music Discovery & Auto-Queue System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 DiscoveryLastFM.py                 # Run normal discovery sync
  python3 DiscoveryLastFM.py --update        # Check and install updates
  python3 DiscoveryLastFM.py --update-status # Show update status
  python3 DiscoveryLastFM.py --list-backups  # List available backups
  python3 DiscoveryLastFM.py --version       # Show current version
        """
    )

    parser.add_argument('--update', action='store_true',
                       help='Check for updates and install if available')
    parser.add_argument('--update-status', action='store_true',
                       help='Show current update status and configuration')
    parser.add_argument('--list-backups', action='store_true',
                       help='List available backup versions')
    parser.add_argument('--version', action='store_true',
                       help='Show current version and exit')
    parser.add_argument('--force-update', action='store_true',
                       help='Force update even after failed attempts')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up temporary files and old backups')

    return parser.parse_args()


# Entry point with CLI support and validation
if __name__ == "__main__":
    import argparse

    try:
        args = parse_cli_args()

        # Handle version command
        if args.version:
            from utils.updater import get_current_version
            print(f"DiscoveryLastFM v{get_current_version()}")
            sys.exit(0)

        # Handle update commands
        if args.update:
            handle_update_command()
            sys.exit(0)

        if args.update_status:
            handle_update_status()
            sys.exit(0)

        if args.list_backups:
            handle_backups_list()
            sys.exit(0)

        if args.cleanup:
            from utils.updater import create_updater_from_config
            config_dict = {k: v for k, v in globals().items() if k.isupper()}
            config_dict["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))
            updater = create_updater_from_config(config_dict)
            updater.cleanup_temp_files()
            print("✅ Cleanup completed")
            sys.exit(0)

        # Check for automatic updates if enabled
        if globals().get('AUTO_UPDATE_ENABLED', False):
            try:
                from utils.updater import create_updater_from_config
                config_dict = {k: v for k, v in globals().items() if k.isupper()}
                config_dict["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))
                updater = create_updater_from_config(config_dict)

                if updater.should_check_for_updates():
                    log.info("Checking for automatic updates...")
                    release_info = updater.check_for_updates()
                    if release_info:
                        log.info(f"Update available: {release_info['version']}. Use --update to install.")
            except Exception as e:
                log.warning(f"Auto-update check failed: {e}")

        # Normal sync operation
        validate_configuration()
        sync()

    except KeyboardInterrupt:
        log.warning("Interrupted.")
    except ConfigurationError as e:
        log.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)
