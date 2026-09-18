from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from utils.cli_helpers import notebook_safe_argv, prompt_for_path, stdin_is_interactive

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pretend_notebook(monkeypatch) -> None:
    """Make the ipykernel probe in cli_helpers see a live kernel."""
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))


def _pretend_shell(monkeypatch) -> None:
    """Make the ipykernel probe in cli_helpers see a plain interpreter."""
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)


class _Stdin:
    def __init__(self, isatty_result: bool | type[BaseException]) -> None:
        self._result = isatty_result

    def isatty(self) -> bool:
        if isinstance(self._result, type) and issubclass(self._result, BaseException):
            raise self._result
        assert isinstance(self._result, bool)
        return self._result


# ---------------------------------------------------------------------------
# notebook_safe_argv
# ---------------------------------------------------------------------------


def test_notebook_safe_argv_returns_explicit_argv(monkeypatch) -> None:
    _pretend_shell(monkeypatch)
    assert notebook_safe_argv(["--gtfs", "feed.zip"]) == ["--gtfs", "feed.zip"]


def test_notebook_safe_argv_copies_rather_than_aliasing() -> None:
    original = ["--flag"]
    returned = notebook_safe_argv(original)
    assert returned == original
    assert returned is not original


def test_notebook_safe_argv_accepts_any_sequence() -> None:
    assert notebook_safe_argv(("--a", "--b")) == ["--a", "--b"]


def test_notebook_safe_argv_empty_list_is_not_treated_as_none(monkeypatch) -> None:
    # [] is a caller asking for "no flags", which must survive even in a shell,
    # where None would instead hand argparse sys.argv[1:].
    _pretend_shell(monkeypatch)
    assert notebook_safe_argv([]) == []


def test_notebook_safe_argv_in_notebook_returns_empty_list(monkeypatch) -> None:
    _pretend_notebook(monkeypatch)
    assert notebook_safe_argv(None) == []


def test_notebook_safe_argv_in_shell_returns_none(monkeypatch) -> None:
    _pretend_shell(monkeypatch)
    assert notebook_safe_argv(None) is None


# ---------------------------------------------------------------------------
# stdin_is_interactive
# ---------------------------------------------------------------------------


def test_stdin_is_interactive_true_in_notebook(monkeypatch) -> None:
    # ipykernel wins before stdin is consulted: a kernel routes input() to a
    # prompt widget even though its stdin is not a tty.
    _pretend_notebook(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _Stdin(False))
    assert stdin_is_interactive() is True


def test_stdin_is_interactive_true_on_a_terminal(monkeypatch) -> None:
    _pretend_shell(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _Stdin(True))
    assert stdin_is_interactive() is True


def test_stdin_is_interactive_false_when_redirected(monkeypatch) -> None:
    _pretend_shell(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _Stdin(False))
    assert stdin_is_interactive() is False


def test_stdin_is_interactive_false_when_stdin_is_none(monkeypatch) -> None:
    _pretend_shell(monkeypatch)
    monkeypatch.setattr(sys, "stdin", None)
    assert stdin_is_interactive() is False


def test_stdin_is_interactive_false_when_stdin_is_closed(monkeypatch) -> None:
    # A closed file raises ValueError from isatty(); pytest's own capture can
    # leave stdin in this state.
    _pretend_shell(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _Stdin(ValueError))
    assert stdin_is_interactive() is False


def test_stdin_is_interactive_false_when_stdin_lacks_isatty(monkeypatch) -> None:
    _pretend_shell(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _Stdin(AttributeError))
    assert stdin_is_interactive() is False


# ---------------------------------------------------------------------------
# prompt_for_path
# ---------------------------------------------------------------------------


def _answers(monkeypatch, *replies: str) -> list[str]:
    """Feed *replies* to input() in order, recording the prompts shown."""
    seen: list[str] = []
    pending = list(replies)

    def fake_input(prompt: str = "") -> str:
        seen.append(prompt)
        return pending.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return seen


def test_prompt_for_path_returns_existing_path(monkeypatch, tmp_path) -> None:
    target = tmp_path / "feed.zip"
    target.write_text("x")
    _answers(monkeypatch, str(target))
    assert prompt_for_path("GTFS: ") == target


def test_prompt_for_path_strips_double_quotes(monkeypatch, tmp_path) -> None:
    # Windows Explorer's "Copy as path" wraps the value in double quotes.
    target = tmp_path / "feed.zip"
    target.write_text("x")
    _answers(monkeypatch, f'"{target}"')
    assert prompt_for_path("GTFS: ") == target


def test_prompt_for_path_strips_single_quotes(monkeypatch, tmp_path) -> None:
    target = tmp_path / "feed.zip"
    target.write_text("x")
    _answers(monkeypatch, f"'{target}'")
    assert prompt_for_path("GTFS: ") == target


def test_prompt_for_path_strips_surrounding_whitespace(monkeypatch, tmp_path) -> None:
    target = tmp_path / "feed.zip"
    target.write_text("x")
    _answers(monkeypatch, f"  {target}  ")
    assert prompt_for_path("GTFS: ") == target


def test_prompt_for_path_blank_returns_default(monkeypatch, tmp_path) -> None:
    default = tmp_path / "never_created.zip"
    _answers(monkeypatch, "")
    assert prompt_for_path("GTFS: ", default=default) == default


def test_prompt_for_path_default_is_not_existence_checked(monkeypatch, tmp_path) -> None:
    # must_exist applies to typed answers only, so a missing default still wins.
    default = tmp_path / "never_created.zip"
    _answers(monkeypatch, "")
    assert prompt_for_path("GTFS: ", must_exist=True, default=default) == default


def test_prompt_for_path_blank_returns_none_when_skippable(monkeypatch) -> None:
    _answers(monkeypatch, "")
    assert prompt_for_path("GTFS: ", allow_skip=True) is None


def test_prompt_for_path_default_beats_allow_skip(monkeypatch, tmp_path) -> None:
    default = tmp_path / "default.zip"
    _answers(monkeypatch, "")
    assert prompt_for_path("GTFS: ", default=default, allow_skip=True) == default


def test_prompt_for_path_reasks_after_blank_when_required(monkeypatch, tmp_path) -> None:
    target = tmp_path / "feed.zip"
    target.write_text("x")
    seen = _answers(monkeypatch, "", str(target))
    assert prompt_for_path("GTFS: ") == target
    assert len(seen) == 2


def test_prompt_for_path_reasks_until_the_path_exists(monkeypatch, tmp_path) -> None:
    target = tmp_path / "feed.zip"
    target.write_text("x")
    seen = _answers(monkeypatch, str(tmp_path / "typo.zip"), str(target))
    assert prompt_for_path("GTFS: ") == target
    assert len(seen) == 2


def test_prompt_for_path_accepts_missing_path_when_not_required(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "output_dir"
    _answers(monkeypatch, str(missing))
    assert prompt_for_path("Output: ", must_exist=False) == missing


def test_prompt_for_path_accepts_a_directory(monkeypatch, tmp_path) -> None:
    _answers(monkeypatch, str(tmp_path))
    assert prompt_for_path("Folder: ") == Path(tmp_path)


def test_prompt_for_path_shows_the_prompt_text(monkeypatch, tmp_path) -> None:
    target = tmp_path / "feed.zip"
    target.write_text("x")
    seen = _answers(monkeypatch, str(target))
    prompt_for_path("Path to GTFS feed: ")
    assert seen == ["Path to GTFS feed: "]


def test_prompt_for_path_warns_before_reasking(monkeypatch, tmp_path, caplog) -> None:
    target = tmp_path / "feed.zip"
    target.write_text("x")
    _answers(monkeypatch, str(tmp_path / "typo.zip"), str(target))
    with caplog.at_level("WARNING"):
        prompt_for_path("GTFS: ")
    assert any("Path not found" in record.message for record in caplog.records)


def test_prompt_for_path_propagates_keyboard_interrupt(monkeypatch) -> None:
    # Documented contract: callers catch this and treat it as "user aborted".
    def interrupt(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prompt_for_path("GTFS: ")


def test_prompt_for_path_propagates_eof(monkeypatch) -> None:
    def eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(EOFError):
        prompt_for_path("GTFS: ")
