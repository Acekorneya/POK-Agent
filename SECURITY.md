# Security policy

POK-Agent controls the Windows desktop, runs commands, and reads files, so
security reports are taken seriously.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately
through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**
([direct link](https://github.com/Acekorneya/POK-Agent/security/advisories/new)).

Include what you found, how to reproduce it, the version (Settings or the
release tag), and the impact you expect. You should get a reply within a week.
Please give us a reasonable time to release a fix before disclosing publicly.

## Scope

In scope, for example:

- ways around the safety boundary: typing into password fields, interacting
  with UAC/secure-desktop or elevated windows, input outside the authorized
  window, or joining calls the user did not ask for;
- bypassing approvals, workspace containment, or the attached-file read-only rule;
- prompt injection from screen text, web pages, or files that makes the agent
  take a dangerous action without approval;
- leaking API keys (stored in Windows Credential Manager) into traces, logs,
  or model requests;
- issues in the release workflow or published binaries.

Out of scope: problems that need an already-compromised machine, a model simply
giving a wrong answer, and actions the user explicitly approved or enabled
through the autonomous policy mode.

## Supported versions

Security fixes go into the latest release. Please update before reporting.
