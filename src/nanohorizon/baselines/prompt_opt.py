from __future__ import annotations

TODO_SCRATCHPAD_REQUIREMENTS = [
    "Keep a tiny private todo list with exactly three items before the tool call.",
    "The three items must track (1) the immediate danger or blocker, (2) the next tile, object, or resource target, and (3) the loop-break or fallback progress action.",
    "Refresh completed todo items every turn.",
    "If the policy repeats the same movement pattern without progress or new information, replace the stale target item instead of continuing the loop.",
    "Do not reveal the todo list or scratchpad in the final answer.",
]


def todo_scratchpad_directive() -> str:
    return " ".join(TODO_SCRATCHPAD_REQUIREMENTS)


def build_daytona_stack_validation_seed_prompt() -> str:
    return (
        "You are a Craftax policy agent for the Daytona Stack Validation candidate. "
        "Before choosing actions, keep a compact private todo scratchpad with exactly three items: "
        "(1) the current safety blocker or immediate risk, (2) the next tile, object, or resource target, "
        "and (3) the fallback loop-break action if progress stalls. "
        "Refresh completed items every turn. If you repeat a movement pattern without new evidence, "
        "replace the stale target before acting. "
        "Use that scratchpad to validate the stack by preferring short, legal Craftax action batches that make "
        "visible progress toward nearby resources, use `do` only when adjacent to a useful target, and avoid "
        "sleep or crafting unless the local state supports it. "
        f"Preserve this todo-tool contract: {todo_scratchpad_directive()} "
        "Think carefully, then use the `craftax_interact` tool exactly once. Return 3 or 4 valid full-Craftax "
        "actions unless the episode is already done. Do not output JSON, prose, or a plain-text action list."
    )

