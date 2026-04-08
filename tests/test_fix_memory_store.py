import pytest

from memory.fix_memory_store import FixMemoryStore


@pytest.mark.anyio
async def test_fix_memory_persists_attempts_and_resolutions(tmp_path):
    store = FixMemoryStore(tmp_path)

    await store.record_attempt(
        "src/users.py",
        errors_text="NameError: user_service is not defined",
        error_hash="abc12345",
        referenced_files=["src/services.py"],
    )
    await store.record_resolution(
        "src/users.py",
        "Imported user_service from src/services.py and updated the call site",
    )

    summary = await store.get_summary("src/users.py")
    await store.flush()
    restored = FixMemoryStore(tmp_path)
    restored_summary = await restored.get_summary("src/users.py")

    assert "abc12345" in summary
    assert "src/services.py" in summary
    assert "updated the call site" in summary
    assert restored_summary == summary
