# LangSmith trace handoff

The assignment requires accessible, inspectable LangSmith runs for one successful case and difficult/failure cases. One public cold-recovery trace has been added below; **remaining link cells stay marked pending until real runs are shared from the LangSmith UI.** Do not submit the local JSON/Markdown examples from `docs/trace_exports/` as if they were LangSmith runs; those files are local illustrations, not remotely captured traces.

Project: [Asterline Assistant](https://smith.langchain.com/o/34ea5156-f488-4661-b611-d4fa3993ddf4/a/asterline-assistant/projects/p/e99ab585-0198-416a-b09f-471470f3df09)

## Create or find real runs

1. Sign in to LangSmith and open the project link above. Confirm the project belongs to the workspace where this app's `LANGSMITH_API_KEY` writes traces.
2. In the repository PowerShell terminal, select the improved graph and allow FakeLLM spans (the fake provider is normally excluded from tracing):

   ```powershell
   $env:ASTERLINE_CONFIG = 'improved'
   $env:ASTERLINE_LLM_PROVIDER = 'fake'
   $env:ASTERLINE_TRACE_FAKE = '1'
   $env:LANGSMITH_TRACING = 'true'
   uv run asterline run
   ```

   The app loads `.env` itself; never paste credentials into commands or chat. If LangSmith is over its trace quota, existing runs can still be inspected, but new spans may not appear. Do not claim a new run exists unless it is visible in the project.

3. Exercise the app with synthetic users only. For a quick research run, use `uv run asterline ask "What is the current cold recovery period, and what older value did it supersede?" --provider fake --config improved --user alice`. For memory continuity, save a harmless synthetic preference in the UI, start a new/resumed session for the same user, and ask about that preference. The saved preference must be shown as user context, not cited as company evidence.
4. Open the run in LangSmith and inspect the full trace tree and input/output for secrets or personal information before sharing. This project has a limited trace quota; prefer existing real traces when they cover the requested behavior rather than generating many duplicates.

## Share a trace publicly

This is the easiest option for an external grader who may not have a LangSmith account:

1. Open a real trace in the LangSmith UI.
2. In the trace Details view, open the top `More` menu and choose `Share`.
3. Copy the public URL and open it in a signed-out/private browser window to verify it is accessible.
4. Paste that URL in the matching row below and in the assignment portal. Anyone with a public trace link can view that trace; only share synthetic, redacted payloads. Use `Settings → Shared URLs` to review or revoke links later.

LangSmith currently documents public trace sharing as `More → Share`; public URLs are viewable by anyone holding the link. See [Manage a trace](https://docs.langchain.com/langsmith/manage-trace) and [LangSmith guidance on public sharing controls](https://support.langchain.com/articles/5345880639-how-do-i-disable-public-sharing-for-traces-in-langsmith).

## Invite the named reviewer (alternative)

If you prefer account-restricted access, an organization/workspace admin can invite `radialpulse@nxtwave.co.in` in LangSmith Settings → Members and roles, selecting membership for the workspace containing this project. The assignment's required option is access to the project or an accessible project link; do not grant organization-admin access. The exact invite controls depend on the account plan and your role. See [LangSmith user management](https://docs.langchain.com/langsmith/user-management) and [organization/workspace setup](https://docs.langchain.com/langsmith/set-up-hierarchy).

## Submission links — fill after sharing

| Required example | Expected inspection | Shared LangSmith URL |
|---|---|---|
| Successful conflict answer | Current value is used and superseded value is marked | [Public cold-recovery trace](https://smith.langchain.com/public/b5703387-07db-4f06-aabf-a5939e92a6c1/r/01a18183-b365-773a-8558-1eb88a5a1edb?start_time=2026-10-28T07%3A57%3A34.437694Z) |
| Memory-aware follow-up after a restart | Only the selected synthetic user's saved context is used and labelled | **PENDING — paste public trace URL** |
| Honest abstention | Missing/unpublished fact is declined without unsupported claims | **PENDING — paste public trace URL** |
| Verifier repair | First draft is repaired once, then checked | **PENDING — paste public trace URL** |
| Genuine provider failure or degraded response | Sanitized error and clean fallback/source-only behavior | **PENDING — paste public trace URL** |

Run `uv run asterline export-traces` to create local redacted JSON/Markdown exports and to attempt best-effort dataset/report upload when credentials are configured. The command does **not** make the local examples into real LangSmith runs and does not create the public links above.

Deleted data may remain in already-shipped traces. The evaluator datasets contain synthetic assignment examples; still inspect every actual trace before sharing it.
