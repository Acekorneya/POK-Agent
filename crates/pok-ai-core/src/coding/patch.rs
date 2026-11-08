//! `apply_patch`: a multi-file edit in the compact patch format coding models
//! already know (`*** Begin Patch` / `*** Update File:` / `@@` / `+` `-`).
//! Every file is checked before any is written, so a patch applies completely
//! or not at all.

use super::*;

const MAX_PATCH_BYTES: usize = 200 * 1024;
const MAX_HUNK_OLD_LINES: usize = 60;

#[derive(Debug, Deserialize, JsonSchema)]
struct ApplyPatchArgs {
    /// The whole patch. Format:
    /// *** Begin Patch
    /// *** Update File: path/to/file
    /// @@ optional line of code that precedes the change
    ///  unchanged context line (one leading space)
    /// -line to remove
    /// +line to add
    /// *** Add File: path/to/new_file
    /// +every line of the new file
    /// *** End Patch
    /// Use two or three unchanged lines around each change so it can be located.
    patch: String,
}

#[derive(Debug, PartialEq)]
enum PatchOp {
    Add { path: String, lines: Vec<String> },
    Update { path: String, hunks: Vec<Hunk> },
}

#[derive(Debug, PartialEq, Default)]
struct Hunk {
    /// Text of the line that precedes the change, from the `@@` header.
    anchor: Option<String>,
    old: Vec<String>,
    new: Vec<String>,
}

fn parse_patch(patch: &str) -> std::result::Result<Vec<PatchOp>, String> {
    let lines = patch.lines().collect::<Vec<_>>();
    let mut index = lines
        .iter()
        .position(|line| line.trim() == "*** Begin Patch")
        .map_or(0, |at| at + 1);
    let mut ops = Vec::new();
    while index < lines.len() {
        let line = lines[index];
        let trimmed = line.trim();
        if trimmed == "*** End Patch" {
            break;
        }
        if trimmed.is_empty() {
            index += 1;
            continue;
        }
        if let Some(path) = trimmed.strip_prefix("*** Add File:") {
            index += 1;
            let mut content = Vec::new();
            while index < lines.len() && !lines[index].starts_with("*** ") {
                let text = lines[index];
                content.push(text.strip_prefix('+').unwrap_or(text).to_owned());
                index += 1;
            }
            ops.push(PatchOp::Add {
                path: path.trim().to_owned(),
                lines: content,
            });
        } else if let Some(path) = trimmed.strip_prefix("*** Update File:") {
            index += 1;
            let mut hunks = Vec::new();
            let mut current: Option<Hunk> = None;
            while index < lines.len() && !lines[index].starts_with("*** ") {
                let text = lines[index];
                if let Some(header) = text.strip_prefix("@@") {
                    if let Some(done) = current.take() {
                        hunks.push(done);
                    }
                    let header = header.trim().trim_end_matches("@@").trim();
                    current = Some(Hunk {
                        anchor: (!header.is_empty()).then(|| header.to_owned()),
                        ..Hunk::default()
                    });
                } else {
                    let hunk = current.get_or_insert_with(Hunk::default);
                    match text.chars().next() {
                        Some('+') => hunk.new.push(text[1..].to_owned()),
                        Some('-') => hunk.old.push(text[1..].to_owned()),
                        Some(' ') => {
                            hunk.old.push(text[1..].to_owned());
                            hunk.new.push(text[1..].to_owned());
                        }
                        // A blank context line whose leading space was dropped.
                        None => {
                            hunk.old.push(String::new());
                            hunk.new.push(String::new());
                        }
                        Some(_) => {
                            return Err(format!(
                                "patch line {} must start with ' ', '-', '+' or '@@': {text:?}",
                                index + 1
                            ));
                        }
                    }
                }
                index += 1;
            }
            if let Some(done) = current.take() {
                hunks.push(done);
            }
            if hunks.is_empty() {
                return Err(format!("no changes listed for {}", path.trim()));
            }
            ops.push(PatchOp::Update {
                path: path.trim().to_owned(),
                hunks,
            });
        } else if trimmed.starts_with("*** Delete File:") || trimmed.starts_with("*** Move to:") {
            return Err(format!(
                "{trimmed:?} is not supported: use run_command to delete or move files"
            ));
        } else if trimmed == "*** End of File" {
            index += 1;
        } else {
            return Err(format!(
                "unexpected line {}: {line:?}; expected *** Add File, *** Update File, or *** End Patch",
                index + 1
            ));
        }
    }
    if ops.is_empty() {
        return Err("the patch changes nothing".into());
    }
    Ok(ops)
}

/// Find `needle` (a block of lines) in `haystack` at or after `from`:
/// exactly, then ignoring trailing whitespace, then ignoring all
/// leading/trailing whitespace.
fn find_block(haystack: &[String], needle: &[String], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    let last = haystack.len() - needle.len();
    let normalizers: [fn(&str) -> String; 3] = [
        |line| line.to_owned(),
        |line| line.trim_end().to_owned(),
        |line| line.trim().to_owned(),
    ];
    for normalize in normalizers {
        let wanted = needle
            .iter()
            .map(|line| normalize(line))
            .collect::<Vec<_>>();
        if let Some(at) = (from..=last).find(|start| {
            haystack[*start..*start + wanted.len()]
                .iter()
                .zip(&wanted)
                .all(|(line, expected)| &normalize(line) == expected)
        }) {
            return Some(at);
        }
    }
    None
}

/// Apply the hunks to a file's text; the error says which hunk failed and why.
fn apply_hunks(path: &str, text: &str, hunks: &[Hunk]) -> std::result::Result<String, String> {
    let crlf = text.contains("\r\n");
    let normalized = text.replace("\r\n", "\n");
    let trailing_newline = normalized.ends_with('\n');
    let mut lines = normalized
        .strip_suffix('\n')
        .unwrap_or(&normalized)
        .split('\n')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if normalized.is_empty() {
        lines.clear();
    }
    let mut cursor = 0;
    for (number, hunk) in hunks.iter().enumerate() {
        if hunk.old.len() > MAX_HUNK_OLD_LINES {
            return Err(format!(
                "hunk {} for {path} removes or matches {} lines; keep each hunk to at most {MAX_HUNK_OLD_LINES} and split the change",
                number + 1,
                hunk.old.len()
            ));
        }
        // The `@@` line moves the search start to the line it names.
        if let Some(anchor) = &hunk.anchor
            && let Some(at) = lines[cursor.min(lines.len())..]
                .iter()
                .position(|line| line.trim() == anchor.trim() || line.contains(anchor.as_str()))
        {
            cursor += at;
        }
        let at = if hunk.old.is_empty() {
            // Pure insertion: after the anchor line, else at the end of the file.
            if hunk.anchor.is_some() && cursor < lines.len() {
                cursor + 1
            } else {
                lines.len()
            }
        } else {
            find_block(&lines, &hunk.old, cursor).ok_or_else(|| {
                let first = hunk
                    .old
                    .iter()
                    .map(|line| line.trim())
                    .find(|line| !line.is_empty())
                    .unwrap_or_default();
                let where_hint = lines
                    .iter()
                    .position(|line| line.trim() == first)
                    .map(|index| {
                        format!(
                            " Its first line appears at line {}; the file there reads:\n{}",
                            index + 1,
                            lines[index..lines.len().min(index + 4)].join("\n")
                        )
                    })
                    .unwrap_or_else(|| {
                        " Its first line is not in the file; read_file the region again.".into()
                    });
                format!(
                    "hunk {} for {path} does not match the file.{where_hint}",
                    number + 1
                )
            })?
        };
        lines.splice(at..at + hunk.old.len(), hunk.new.iter().cloned());
        cursor = at + hunk.new.len();
    }
    let mut result = lines.join("\n");
    if trailing_newline || lines.is_empty() {
        result.push('\n');
    }
    if crlf {
        result = result.replace('\n', "\r\n");
    }
    Ok(result)
}

/// A path for a file that may not exist yet (nor its folders): the nearest
/// existing folder must be inside the authorized workspace, and the rest may
/// not climb out with `..`.
fn resolve_new_path(context: &ToolContext, path: &str) -> Result<PathBuf> {
    let requested = PathBuf::from(path);
    if requested
        .components()
        .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(PokError::Tool(format!("{path} may not contain '..'")));
    }
    let mut existing = if requested.is_absolute() {
        requested
    } else {
        context.workspace.join(requested)
    };
    let mut tail = Vec::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .map(std::ffi::OsStr::to_owned)
            .ok_or_else(|| PokError::Tool(format!("{path} is not a usable path")))?;
        tail.push(name);
        existing = existing
            .parent()
            .ok_or_else(|| PokError::Tool(format!("{path} is not a usable path")))?
            .to_path_buf();
    }
    let mut resolved = context.resolve_workspace_path(existing)?;
    for name in tail.into_iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

pub(super) struct ApplyPatchTool;

#[async_trait]
impl Tool for ApplyPatchTool {
    fn name(&self) -> &'static str {
        "apply_patch"
    }
    fn description(&self) -> &'static str {
        "Edit one or more files with a single patch that applies completely or not at all. Wrap it in `*** Begin Patch` / `*** End Patch`; per file use `*** Update File: path` with `@@` hunks (lines start with ' ' for context, '-' to remove, '+' to add) or `*** Add File: path` with every line prefixed '+'. Give two or three unchanged lines around each change; indentation and line endings are matched leniently. Prefer this over several edit_file calls when a change touches more than one place or file. Deleting or moving files is not supported here. Each result shows what changed and any syntax errors."
    }
    fn input_schema(&self) -> Value {
        schema::<ApplyPatchArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::WorkspaceWrite
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        let args = serde_json::from_value::<ApplyPatchArgs>(args.clone()).ok()?;
        match parse_patch(&args.patch).ok()?.into_iter().next()? {
            PatchOp::Add { path, .. } | PatchOp::Update { path, .. } => Some(PathBuf::from(path)),
        }
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ApplyPatchArgs = serde_json::from_value(args)?;
        if args.patch.len() > MAX_PATCH_BYTES {
            return Err(PokError::Tool(format!(
                "patch exceeds {MAX_PATCH_BYTES} bytes; split it"
            )));
        }
        let ops = parse_patch(&args.patch).map_err(PokError::Tool)?;
        // Plan every file first; nothing is written until all of them check out.
        struct Planned {
            path: PathBuf,
            before: Option<String>,
            after: String,
        }
        let mut planned: Vec<Planned> = Vec::new();
        for op in &ops {
            match op {
                PatchOp::Add { path, lines } => {
                    let resolved = resolve_new_path(context, path)?;
                    if resolved.exists() || planned.iter().any(|plan| plan.path == resolved) {
                        return Err(PokError::Tool(format!(
                            "{path} already exists; use *** Update File to change it"
                        )));
                    }
                    let mut content = lines.join("\n");
                    content.push('\n');
                    if content.len() as u64 > MAX_FILE_BYTES {
                        return Err(PokError::Tool(format!(
                            "{path} exceeds the file size limit"
                        )));
                    }
                    planned.push(Planned {
                        path: resolved,
                        before: None,
                        after: content,
                    });
                }
                PatchOp::Update { path, hunks } => {
                    let resolved = context.resolve_workspace_path(PathBuf::from(path))?;
                    let earlier = planned.iter().position(|plan| plan.path == resolved);
                    let current = match earlier {
                        Some(index) => planned[index].after.clone(),
                        None => std::fs::read_to_string(&resolved).map_err(|error| {
                            PokError::Tool(format!("cannot read {path}: {error}"))
                        })?,
                    };
                    let after = apply_hunks(path, &current, hunks).map_err(PokError::Tool)?;
                    match earlier {
                        Some(index) => planned[index].after = after,
                        None => planned.push(Planned {
                            path: resolved,
                            before: Some(current),
                            after,
                        }),
                    }
                }
            }
        }
        let mut files = Vec::new();
        for plan in planned {
            if let Some(parent) = plan.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            atomic_write(&plan.path, plan.after.as_bytes())?;
            let sha256 = hash_text(&plan.after);
            let (operation, undo_path) = match &plan.before {
                Some(before) => {
                    let record = UndoRecord {
                        path: plan.path.clone(),
                        before: before.clone(),
                        after_hash: sha256.clone(),
                    };
                    let undo = write_undo_record(context, &record)?;
                    if let Some(owned) = context.session_files.lock().get_mut(&plan.path) {
                        owned.sha256 = sha256.clone();
                    }
                    ("edited", Some(undo))
                }
                None => {
                    context.session_files.lock().insert(
                        plan.path.clone(),
                        SessionFileRecord {
                            sha256: sha256.clone(),
                            created_by_session: true,
                        },
                    );
                    ("created", None)
                }
            };
            context
                .artifact_evidence
                .lock()
                .push(inspect_artifact_path(&plan.path, operation)?);
            let diagnostics = diagnostics::check_file(&plan.path).await;
            files.push(json!({
                "path": plan.path,
                "operation": operation,
                "new_sha256": sha256,
                "undo_record": undo_path,
                "change": plan.before.as_deref().map(|before| change_summary(before, &plan.after)),
                "diagnostics": diagnostics::report_value(diagnostics.as_ref()),
            }));
        }
        Ok(json!({"files": files}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_patch_format_is_parsed_into_files_and_hunks() {
        let patch = "*** Begin Patch\n*** Update File: a.txt\n@@ fn main\n keep\n-old\n+new\n+extra\n*** Add File: b/c.txt\n+one\n+two\n*** End Patch";
        let ops = parse_patch(patch).unwrap();
        assert_eq!(ops.len(), 2);
        let PatchOp::Update { path, hunks } = &ops[0] else {
            panic!("update first")
        };
        assert_eq!(path, "a.txt");
        assert_eq!(hunks[0].anchor.as_deref(), Some("fn main"));
        assert_eq!(hunks[0].old, ["keep", "old"]);
        assert_eq!(hunks[0].new, ["keep", "new", "extra"]);
        assert!(parse_patch("*** Begin Patch\n*** Delete File: a\n*** End Patch").is_err());
        assert!(parse_patch("nothing here").is_err());
    }

    #[test]
    fn hunks_apply_leniently_and_keep_line_endings() {
        let text = "fn a() {\r\n    one();\r\n}\r\n\r\nfn b() {\r\n    two();\r\n}\r\n";
        let hunks = vec![Hunk {
            anchor: Some("fn b() {".into()),
            // Indentation differs from the file: still located.
            old: vec!["two();".into()],
            new: vec!["    three();".into()],
        }];
        let result = apply_hunks("x.rs", text, &hunks).unwrap();
        assert_eq!(
            result,
            "fn a() {\r\n    one();\r\n}\r\n\r\nfn b() {\r\n    three();\r\n}\r\n"
        );
        // A hunk that does not match says where the block starts.
        let miss = vec![Hunk {
            old: vec!["one();".into(), "wrong".into()],
            new: vec![],
            anchor: None,
        }];
        let error = apply_hunks("x.rs", text, &miss).unwrap_err();
        assert!(error.contains("first line appears at line 2"), "{error}");
        // Insertion without removal goes after the anchor.
        let insert = vec![Hunk {
            anchor: Some("fn a() {".into()),
            old: vec![],
            new: vec!["    // note".into()],
        }];
        let result = apply_hunks("x.rs", text, &insert).unwrap();
        assert!(
            result.starts_with("fn a() {\r\n    // note\r\n    one();"),
            "{result}"
        );
    }

    #[tokio::test]
    async fn a_patch_applies_to_several_files_or_to_none() {
        let temp = tempfile::tempdir().unwrap();
        let platform = crate::platform::MockDesktop::new(
            crate::builtins::tests::scroll_observation(&[], "patch"),
        );
        let context = crate::builtins::tests::mock_input_context(platform, &temp);
        std::fs::write(temp.path().join("a.txt"), "one\ntwo\nthree\n").unwrap();
        std::fs::write(temp.path().join("b.json"), "{\"a\": 1}\n").unwrap();
        let mut registry = crate::tool::ToolRegistry::new();
        register_coding_tools(&mut registry);
        // The second file's hunk does not match: nothing is written.
        let bad = "*** Begin Patch\n*** Update File: a.txt\n two\n-three\n+3\n*** Update File: b.json\n-nope\n+x\n*** End Patch";
        let error = registry
            .call("apply_patch", json!({"patch": bad}), &context)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("b.json"), "{error}");
        assert_eq!(
            std::fs::read_to_string(temp.path().join("a.txt")).unwrap(),
            "one\ntwo\nthree\n"
        );
        // A good patch changes both files, adds one, and reports diagnostics.
        let good = "*** Begin Patch\n*** Update File: a.txt\n two\n-three\n+3\n*** Update File: b.json\n-{\"a\": 1}\n+{\"a\": 2,}\n*** Add File: sub/new.txt\n+hello\n*** End Patch";
        let result = registry
            .call("apply_patch", json!({"patch": good}), &context)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(temp.path().join("a.txt")).unwrap(),
            "one\ntwo\n3\n"
        );
        assert_eq!(
            std::fs::read_to_string(temp.path().join("sub/new.txt")).unwrap(),
            "hello\n"
        );
        let files = result["files"].as_array().unwrap();
        assert_eq!(files.len(), 3);
        // The trailing comma made the JSON invalid, and the result says so.
        assert_eq!(files[1]["diagnostics"]["syntax"], "errors", "{result}");
        // An existing file cannot be "added" over.
        let again = "*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch";
        assert!(
            registry
                .call("apply_patch", json!({"patch": again}), &context)
                .await
                .is_err()
        );
    }
}
