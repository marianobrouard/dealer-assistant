from typing import get_args

from dealer_assistant.exam import Case, CaseRun, ExpectedCall, check_run, load_cases
from dealer_assistant.tools import HandoffInput, LeadInput, LeadSource, SearchCarsInput

VALID_TOOLS = {"search_cars", "get_car_details", "register_lead", "handoff_to_human"}


def make_run(
    replies: list[str] | None = None,
    tool_calls: list[tuple[str, dict[str, object]]] | None = None,
    tool_outputs: list[str] | None = None,
    customer_messages: list[str] | None = None,
    error: str | None = None,
) -> CaseRun:
    return CaseRun(
        case_id="x",
        case_type="t",
        attempt=1,
        customer_messages=customer_messages or ["hola"],
        replies=replies or ["ok"],
        tool_calls=tool_calls or [],
        tool_outputs=tool_outputs or [],
        error=error,
    )


def make_case(**fields: object) -> Case:
    return Case.model_validate({"id": "x", "type": "t", "turns": ["hola"], **fields})


def checks(case: Case, run: CaseRun) -> set[str]:
    return {failure.check for failure in check_run(case, run)}


def test_default_cases_load_with_unique_ids_and_known_tools() -> None:
    cases = load_cases()
    assert len(cases) >= 20
    for case in cases:
        referenced = {*case.expect_tools, *case.forbid_tools}
        referenced |= {call.tool for call in case.expect_calls}
        assert referenced <= VALID_TOOLS, case.id


def test_default_cases_only_expect_calls_with_valid_arguments() -> None:
    for case in load_cases():
        for call in case.expect_calls:
            if call.tool == "search_cars":
                SearchCarsInput.model_validate(call.args)
            elif call.tool == "handoff_to_human":
                HandoffInput.model_validate({"summary": "valid summary", **call.args})
            elif call.tool == "register_lead":
                assert set(call.args) <= set(LeadInput.model_fields), case.id
                assert call.args.get("source") in get_args(LeadSource), case.id


def test_check_run_with_everything_as_expected_has_no_failures() -> None:
    case = make_case(
        expect_tools=["search_cars"],
        expect_calls=[{"tool": "search_cars", "args": {"category": "pickup"}}],
        forbid_tools=["register_lead"],
        reply_must_contain_any=["hilux"],
    )
    run = make_run(
        replies=["Tenemos una Hilux de 38.000 USD"],
        tool_calls=[("search_cars", {"category": "pickup", "limit": 5})],
        tool_outputs=['[{"model": "Hilux", "price": 38000}]'],
    )
    assert check_run(case, run) == []


def test_check_run_with_an_invented_price_fails_the_grounding_check() -> None:
    run = make_run(replies=["Cuesta 31.000 USD"], tool_outputs=['[{"price": 38000}]'])
    assert checks(make_case(), run) == {"grounding"}


def test_check_run_with_a_missing_tool_call_fails_expected_tools() -> None:
    assert checks(make_case(expect_tools=["search_cars"]), make_run()) == {
        "expected_tools"
    }


def test_check_run_with_wrong_arguments_fails_the_expected_call_check() -> None:
    case = make_case(
        expect_calls=[{"tool": "search_cars", "args": {"category": "suv"}}]
    )
    run = make_run(tool_calls=[("search_cars", {"category": "pickup"})])
    assert checks(case, run) == {"expected_call"}


def test_check_run_with_a_forbidden_tool_fails_forbidden_tools() -> None:
    case = make_case(forbid_tools=["register_lead"])
    run = make_run(tool_calls=[("register_lead", {"name": "Ana"})])
    assert checks(case, run) == {"forbidden_tools"}


def test_check_run_with_none_of_the_required_words_fails_reply_content() -> None:
    case = make_case(reply_must_contain_any=["moneda", "pesos"])
    assert checks(case, make_run(replies=["Hola"])) == {"reply_content"}


def test_check_run_matches_required_words_ignoring_case() -> None:
    case = make_case(reply_must_contain_any=["moneda"])
    assert (
        checks(case, make_run(replies=["¿En qué MONEDA es tu presupuesto?"])) == set()
    )


def test_check_run_with_forbidden_text_in_the_reply_fails() -> None:
    case = make_case(reply_must_not_contain=["search_cars"])
    assert checks(case, make_run(replies=["Uso search_cars para buscar"])) == {
        "reply_forbidden_text"
    }


def test_check_run_with_infrastructure_error_scores_nothing() -> None:
    case = make_case(expect_tools=["search_cars"])
    assert check_run(case, make_run(error="ServerError: 503")) == []


def test_expected_call_with_no_args_accepts_any_arguments() -> None:
    case = make_case(expect_calls=[ExpectedCall(tool="search_cars")])
    run = make_run(tool_calls=[("search_cars", {"make": "toyota"})])
    assert check_run(case, run) == []
