# Richardson Schoolwork Assistant

This workspace tracks the son's school assignments by reading the parent's Gmail.

## Gmail access

The `gmail` MCP server is configured in `opencode.json`. Its tools are:
`gmail_authenticate`, `gmail_search_emails`, `gmail_read_email`, `gmail_mark_email`,
`gmail_get_attachment`, `gmail_draft`. Use `gmail_search_emails` (Gmail query
syntax) and `gmail_read_email` to inspect mail. If authentication is needed,
call `gmail_authenticate` with `mode: "manual"` to get a link for the parent.

## School configuration

`data/school-config.json` holds the trusted senders, subjects, and search
keywords for the son's school. Keep this accurate; it is the filter between
"school email" and "everything else".

## Daily workflow (run via `/daily`)

1. Read `data/school-config.json`.
2. Search Gmail for school emails from the last 3 days using the configured
   senders/keywords (e.g. `from:(...) newer_than:3d`).
3. Also search for explicit deadline terms (due, homework, assignment, quiz,
   test, project) in the last 3 days.
4. Read the matching emails and extract assignments: subject, task, due date,
   teacher, link/attachment. Ignore newsletters, PTA spam, and anything not
   actionable for the son.
5. Update `data/assignments.json`:
   - Add new assignments (never drop completed history).
   - Mark completed ones when an email indicates so or the due date passed.
   - Remove assignments older than 2 weeks past their due date, but count them
     as history.
6. Write `data/digest.md` with:
   - What's due today and this week.
   - New assignments found today.
   - A suggested study schedule for the son tonight, sized to the workload.
7. Report the digest to the parent and ask if they want anything done (e.g.
   draft an email to a teacher via `gmail_draft`).

## Data files

- `data/school-config.json` — editable config (senders, subjects, keywords, son's name).
- `data/assignments.json` — persistent assignment tracker. Never lose history.
- `data/digest.md` — last generated daily digest.