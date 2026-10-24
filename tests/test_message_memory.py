from backend.service import message_memory
from backend.service.message_memory import COMPRESSION_THRESHOLD, KEEP_RECENT_MESSAGES, message_memory_optimize


def _messages(count):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(count)]


def test_compresses_long_history(temp_session_db, monkeypatch):
    monkeypatch.setattr(message_memory, "generate_summary", lambda messages, previous_summary: "SUMMARY")
    message_memory_optimize("s", _messages(2))
    result = message_memory_optimize("s", _messages(COMPRESSION_THRESHOLD + 3))
    assert result[0]["content"].endswith("SUMMARY")
    assert result[1:] == _messages(COMPRESSION_THRESHOLD + 3)[-KEEP_RECENT_MESSAGES:]


def test_failed_summary_keeps_history(temp_session_db, monkeypatch):
    monkeypatch.setattr(message_memory, "generate_summary", lambda messages, previous_summary: None)
    message_memory_optimize("s", _messages(2))
    history = _messages(COMPRESSION_THRESHOLD + 3)
    assert message_memory_optimize("s", history) == history
    assert temp_session_db.get_session_message("s")["index"] == 0


def test_cleared_history_resets_summary(temp_session_db, monkeypatch):
    monkeypatch.setattr(message_memory, "generate_summary", lambda messages, previous_summary: "SUMMARY")
    message_memory_optimize("s", _messages(2))
    message_memory_optimize("s", _messages(COMPRESSION_THRESHOLD + 3))

    # The client cleared the conversation but kept the session id
    fresh = _messages(1)
    assert message_memory_optimize("s", fresh) == fresh
    record = temp_session_db.get_session_message("s")
    assert record["index"] == 0
    assert record["summary"] is None
