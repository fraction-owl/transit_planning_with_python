"""Tests for the exit_code_contract check.

CONTRIBUTING.md pins `0` success / `1` runtime failure / `2` configuration
error, returned rather than raised so `main()` stays traceback-free in a
notebook cell. This check grades the three ways that breaks statically: a
`main()` not annotated `-> int`, a bare `return` inside it, and a `__main__`
guard that drops the value.

Each rule is covered in both directions, and the guard shapes actually in use
across `scripts/` are pinned as passing, so tightening the check later cannot
start flagging them without failing here.
"""

from __future__ import annotations

from pathlib import Path

from dev_tools.check_style import check_exit_code_contract

DUMMY = Path("example.py")

GOOD_MAIN = "def main(argv: list[str] | None = None) -> int:\n    return 0\n"


def _module(main_body: str, guard: str) -> str:
    return f'"""Summary."""\n\n{main_body}\n\n{guard}'


# =============================================================================
# the shapes in use across scripts/ must keep passing
# =============================================================================
# Four scripts stage the code in a local before raising, so a notebook kernel
# never sees a SystemExit traceback. All of them honour the contract.


def test_canonical_guard_passes() -> None:
    src = _module(GOOD_MAIN, 'if __name__ == "__main__":\n    raise SystemExit(main())\n')
    assert check_exit_code_contract(src, DUMMY) == []


def test_guard_via_local_variable_passes() -> None:
    src = _module(
        GOOD_MAIN,
        'if __name__ == "__main__":\n'
        "    exit_code = main()\n"
        '    if "ipykernel" not in sys.modules:\n'
        "        raise SystemExit(exit_code)\n",
    )
    assert check_exit_code_contract(src, DUMMY) == []


def test_guard_via_try_except_passes() -> None:
    src = _module(
        GOOD_MAIN,
        'if __name__ == "__main__":\n'
        "    try:\n"
        "        exit_code = main()\n"
        "    except Exception:\n"
        "        exit_code = 1\n"
        "    raise SystemExit(exit_code)\n",
    )
    assert check_exit_code_contract(src, DUMMY) == []


def test_guard_calling_a_cli_wrapper_passes() -> None:
    src = _module(
        "def main(config_file: str = CONFIG_PATH) -> int:\n    return 0\n",
        'if __name__ == "__main__":\n'
        "    if launched_by_notebook():\n"
        "        main(CONFIG_PATH)\n"
        "    else:\n"
        "        raise SystemExit(cli())\n",
    )
    assert check_exit_code_contract(src, DUMMY) == []


def test_sys_exit_guard_passes() -> None:
    src = _module(GOOD_MAIN, 'if __name__ == "__main__":\n    sys.exit(main())\n')
    assert check_exit_code_contract(src, DUMMY) == []


# =============================================================================
# main() must be annotated -> int
# =============================================================================


def test_main_annotated_none_is_flagged() -> None:
    src = _module(
        "def main() -> None:\n    run()\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert found[0].check == "exit_code_contract"
    assert found[0].line == 3
    assert "-> None" in found[0].message


def test_main_with_no_annotation_is_flagged() -> None:
    src = _module(
        "def main():\n    return 0\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert "-> None" in found[0].message


# =============================================================================
# no bare return inside main()
# =============================================================================


def test_bare_return_in_main_is_flagged() -> None:
    src = _module(
        "def main() -> int:\n    if not CONFIGURED:\n        return\n    return 0\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert found[0].line == 5
    assert "bare `return`" in found[0].message


def test_explicit_return_none_in_main_is_flagged() -> None:
    src = _module(
        "def main() -> int:\n    return None\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert "bare `return`" in found[0].message


def test_bare_return_in_a_helper_is_not_main_s_problem() -> None:
    src = _module(
        "def helper() -> None:\n    return\n\n\ndef main() -> int:\n    helper()\n    return 0\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    assert check_exit_code_contract(src, DUMMY) == []


def test_bare_return_in_a_closure_inside_main_is_not_flagged() -> None:
    # A nested def has its own scope, so its `return` is not main()'s exit code.
    # scripts/service_coverage/site_route_proximity_arcpy.py defines one.
    src = _module(
        "def main() -> int:\n"
        "    def _norm(value: str) -> None:\n"
        "        if not value:\n"
        "            return\n"
        "        store(value)\n"
        "\n"
        "    _norm(CONFIG)\n"
        "    return 0\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    assert check_exit_code_contract(src, DUMMY) == []


def test_bare_return_in_a_lambda_inside_main_is_not_flagged() -> None:
    src = _module(
        "def main() -> int:\n    rows.sort(key=lambda r: r.name)\n    return 0\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    assert check_exit_code_contract(src, DUMMY) == []


def test_bare_return_inside_a_with_or_try_in_main_is_still_flagged() -> None:
    # Nested *blocks* are main's own scope, unlike nested functions.
    src = _module(
        "def main() -> int:\n"
        "    try:\n"
        "        with open(PATH) as handle:\n"
        "            if not handle.read():\n"
        "                return\n"
        "    except OSError:\n"
        "        return 1\n"
        "    return 0\n",
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert "bare `return`" in found[0].message


# =============================================================================
# the guard must propagate the value
# =============================================================================


def test_bare_main_call_guard_is_flagged() -> None:
    src = _module(GOOD_MAIN, 'if __name__ == "__main__":\n    main()\n')
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert found[0].line == 7
    assert "discards main()'s return value" in found[0].message


def test_raise_systemexit_without_a_value_is_flagged() -> None:
    src = _module(GOOD_MAIN, 'if __name__ == "__main__":\n    main()\n    raise SystemExit\n')
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert "discards main()'s return value" in found[0].message


def test_reversed_guard_comparison_is_still_recognised() -> None:
    src = _module(GOOD_MAIN, 'if "__main__" == __name__:\n    main()\n')
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 1
    assert "discards main()'s return value" in found[0].message


# =============================================================================
# scope: absent pieces belong to the other checks
# =============================================================================


def test_missing_guard_is_left_to_notebook_guard() -> None:
    assert check_exit_code_contract(_module(GOOD_MAIN, "X = 1\n"), DUMMY) == []


def test_missing_main_is_left_to_main_function() -> None:
    src = '"""Summary."""\n\nif __name__ == "__main__":\n    raise SystemExit(run())\n'
    assert check_exit_code_contract(src, DUMMY) == []


def test_both_rules_can_fire_on_one_file() -> None:
    src = _module("def main() -> None:\n    run()\n", 'if __name__ == "__main__":\n    main()\n')
    found = check_exit_code_contract(src, DUMMY)
    assert len(found) == 2
    assert {"-> None" in v.message for v in found} == {True, False}


def test_syntax_error_is_skipped() -> None:
    assert check_exit_code_contract('"""Summary."""\n\ndef broken(:\n', DUMMY) == []
