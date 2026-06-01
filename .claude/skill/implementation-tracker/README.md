# implementation-tracker

A [Claude Code](https://claude.com/claude-code) skill that maintains checkbox-based
progress tracking for multi-session implementation work. It keeps a long-running
implementation plan in sync with what's actually been built, so Claude can pick up
where a prior session left off without duplicating work or losing context.

## What it does

When you're working from a large implementation plan across several sessions, this
skill teaches Claude to:

- **Resume reliably** — read the tracker before writing any code, then report how
  many tasks are done, what was last completed, and what's next.
- **Track at the right granularity** — one checkbox per discrete, verifiable unit
  of work (roughly 15 minutes to a few hours each).
- **Update as it goes** — mark tasks complete before moving on, note in-progress
  and blocked work inline, and preserve history when the plan changes.

## When it triggers

The skill activates whenever you're working from an implementation plan or resuming
ongoing coding work — for example:

- "Continue where we left off"
- "What's left to do?" / "What's next?"
- "Pick up from yesterday"
- Any session where a `PLAN.md`, `IMPLEMENTATION_PLAN.md`, or
  `IMPLEMENTATION_TASKS.md` is present, even if you don't mention it.

## How it works

1. **Session Start Protocol** — Claude first looks for a tracking file (inline
   checkboxes in `IMPLEMENTATION_PLAN.md` / `PLAN.md` / `docs/IMPLEMENTATION_PLAN.md`,
   or a standalone `IMPLEMENTATION_TASKS.md`). If it finds one, it summarizes
   progress before doing anything else. If it doesn't, it asks where the plan is
   rather than guessing.
2. **Setting up tracking** — If the plan already has checkboxes, it's used in place.
   If not, Claude creates an `IMPLEMENTATION_TASKS.md` and extracts tasks from the
   plan at consistent granularity.
3. **Updating checkboxes** — Tasks are marked `[x]` only when done, updated one at a
   time, and the history of completed and changed work is preserved.

See [SKILL.md](SKILL.md) for the full instructions Claude follows.

## Installation

Place this skill where Claude Code looks for skills:

```bash
# Personal skills (available across all projects)
git clone https://github.com/Ronenii/implementation-tracker.git \
  ~/.claude/skills/implementation-tracker

# Or scoped to a single project
git clone https://github.com/Ronenii/implementation-tracker.git \
  .claude/skills/implementation-tracker
```

Once installed, the skill loads automatically when its trigger conditions are met.

## License

[MIT](LICENSE) © 2026 Ronen Gelmanovich
