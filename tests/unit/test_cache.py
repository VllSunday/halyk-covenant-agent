from pathlib import Path

import pytest

from halyk.llm.cache import CacheMissError, CachePolicy, ModelCache

CALL = {
    "model": "gpt-5.6-sol",
    "params": {"temperature": 0.0},
    "system_prompt": "компилируй ковенант",
    "payload": {"clause": "оборот не менее 50 млн"},
}


def make_cache(tmp_path: Path, policy: CachePolicy) -> ModelCache:
    return ModelCache(directory=tmp_path / "model_cache", policy=policy)


def test_key_ignores_dict_order(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, CachePolicy.READ_WRITE)
    straight = cache.key(**CALL, source_hashes=("a", "b"))
    shuffled = cache.key(**CALL, source_hashes=("b", "a"))
    assert straight == shuffled


def test_key_depends_on_sources(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, CachePolicy.READ_WRITE)
    assert cache.key(**CALL, source_hashes=("a",)) != cache.key(**CALL, source_hashes=("b",))


def test_write_only_never_reads(tmp_path: Path) -> None:
    # Ключевое место политики: новый прогон не должен подхватить прошлые ответы.
    cache = make_cache(tmp_path, CachePolicy.WRITE_ONLY)
    key = cache.key(**CALL)
    cache.put(key, {"text": "ответ"})
    assert cache.get(key) is None


def test_read_write_returns_stored(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, CachePolicy.READ_WRITE)
    key = cache.key(**CALL)
    cache.put(key, {"text": "ответ"})
    assert cache.get(key) == {"text": "ответ"}
    assert cache.stats.hits == 1
    assert cache.stats.live == 1


def test_replay_miss_is_an_error(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, CachePolicy.REPLAY)
    with pytest.raises(CacheMissError):
        cache.get(cache.key(**CALL))
