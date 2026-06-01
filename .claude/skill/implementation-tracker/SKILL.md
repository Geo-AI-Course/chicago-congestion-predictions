---
name: implementation-tracker
description: >
  Maintains checkbox-based progress tracking for multi-session implementation work.
  Use this skill whenever the user is working from an implementation plan, resuming
  coding work from a previous session, or needs to track task completion across
  sessions. Trigger on phrases like "continue where we left off", "what's left to
  do", "pick up from yesterday", "what's next", or any session that references an
  existing plan, codebase, or ongoing feature build. Also trigger at the start of
  any session where a PLAN.md, IMPLEMENTATION_PLAN.md, or IMPLEMENTATION_TASKS.md
  is present — even if the user doesn't mention it explicitly.
---

# Implementation Tracker

**Do not write any code or make any changes until you have completed the Session
Start Protocol below.** Reading the tracker first is not optional, it's how you
avoid duplicating work or missing context from a prior session.

---

## Session Start Protocol

1. Search for a tracking file in this order:
   - Inline checkboxes in `IMPLEMENTATION_PLAN.md`, `PLAN.md`, or
     `docs/IMPLEMENTATION_PLAN.md`
   - A standalone `IMPLEMENTATION_TASKS.md` in the project root

2. **If found:** Read it, then tell the user:
   - How many tasks are done vs. total (e.g., "7 of 23 tasks complete")
   - What was last completed
   - What's next

3. **If not found:** Ask the user where the plan is before doing anything else.
   Don't guess, don't start implementing, don't create a tracker from scratch
   without a source plan.

---

## Setting Up Tracking

**If the plan already has checkboxes** — use it in place. Don't create a
duplicate file.

**If the plan has no checkboxes** — create `IMPLEMENTATION_TASKS.md` in the
project root. Extract tasks from the plan at consistent granularity: one checkbox
per discrete, verifiable unit of work. See Granularity section below.

```markdown
# Implementation Tasks

Source: [filename or path of original plan]
Last updated: YYYY-MM-DD

## Phase 1: [Name]

- [ ] Task description
- [ ] Task description
  - [ ] Sub-task (only if genuinely distinct from parent)

## Phase 2: [Name]

...
```

---

## Updating Checkboxes

- Mark `[x]` only when the task is done by your own definition of done —
  whatever that means for this project (tests passing, manually verified etc.)
- For in-progress work: `- [ ] Task <!-- in progress: done X, blocked on Y -->`
- Update the file **before moving to the next task**, not in bulk at the end
- If a task needs splitting mid-implementation, add sub-items rather than
  rewriting the parent
- If the plan changes, add new tasks with `<!-- added YYYY-MM-DD -->` rather
  than deleting completed ones

---

## Task Granularity

Wrong level of detail breaks the tracker in opposite directions:

| Too coarse | Too fine | Right |
|---|---|---|
| "Implement authentication" | "Write opening brace of login function" | "Add POST /auth/login endpoint" |
| "Build the dashboard" | "Name the css class" | "Create DashboardLayout component with responsive grid" |

Aim for tasks that take roughly 15 minutes to a few hours each — independently
completable and verifiable.

---

## End State

When all checkboxes are marked complete:
1. Tell the user the plan is fully implemented
2. Note the completion date in the tracking file header
3. Leave the file in place — it's a useful record of what was built

---

## Edge Cases

- **Ambiguous prior completion:** If unsure whether a prior session finished
  something, check the code before marking it done.
- **Blocked task:** Note the blocker inline rather than skipping or marking
  done. `<!-- blocked: waiting on API key from client -->`
- **Plan changes mid-implementation:** Add tasks, don't delete completed ones.
  The history matters.
- **Can't find the plan:** Ask. Don't invent structure or start fresh without
  the user confirming there's no existing plan.