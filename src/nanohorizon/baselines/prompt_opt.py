from __future__ import annotations

TRACK_ID = "prompt_opt_1usd_gpt54_family"
PRIMARY_TOOL_NAME = "craftax_interact"

TODO_SCRATCHPAD_REQUIREMENTS = [
    "Keep a tiny private todo list with exactly three items before the tool call.",
    "The three items must track (1) the most urgent danger or blocker, (2) the next tile, object, or resource target, and (3) the fallback action that breaks a loop if progress stalls.",
    "Refresh completed todo items every turn.",
    "If the policy repeats the same movement pattern without progress or new information, replace the stale target item instead of continuing the loop.",
    "Do not reveal the todo list or scratchpad in the final answer.",
]


def todo_scratchpad_directive() -> str:
    return " ".join(TODO_SCRATCHPAD_REQUIREMENTS)


def build_seed_prompt(candidate_label: str = "Auto Push E2E") -> str:
    return (
        "You are a Craftax policy agent. Before choosing actions, keep a tiny private "
        "todo list with exactly three items: (1) the most urgent danger or blocker, "
        "(2) the next tile, object, or resource you should reach, and (3) the fallback "
        "action that breaks a loop if progress stalls. Refresh completed todo items "
        "every turn. If you repeat the same movement pattern without new progress or "
        "information, replace the stale target item before acting. Do not reveal the todo "
        "list to the user. Prefer early-game progression: move toward nearby trees or other "
        "gatherable resources, use `do` only when adjacent to a useful target, and avoid sleep, "
        "crafting, or inventory-only actions unless the local state clearly supports them. "
        "Choose a short 3 or 4 action batch that follows the first todo item and, when safe, ends "
        "next to a useful target for the next turn. Think carefully, then use the `craftax_interact` "
        "tool exactly once. Return 3 or 4 valid full-Craftax actions unless the episode is already done. "
        "Use only the tool call as the final answer. Do not output JSON, prose, or a plain-text action list. "
        "As an end-to-end handoff guard, keep the scratchpad current even if the first item is already satisfied."
    )


def build_reflection_system_directive() -> str:
    return (
        "You rewrite Craftax system prompts for a tool-calling policy. "
        f"Preserve these hard requirements: the policy must use the `{PRIMARY_TOOL_NAME}` "
        "tool exactly once, must not answer with JSON or a plain-text action list, and must "
        f"preserve this todo-tool contract: {todo_scratchpad_directive()} Return only the "
        "revised prompt text."
    )

