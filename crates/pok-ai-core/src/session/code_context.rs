//! Code context for coding tasks, supplied before the planner's first call so
//! it does not spend turns on `ls`, `git status`, `grep` and `cat`: a brief of
//! the workspace (layout, git state, how tests run) and the code regions the
//! request names (definitions and the tests that mention them).

use std::path::Path;

use super::*;

const MAX_FILES_SCANNED: usize = 4_000;
const MAX_FILE_BYTES: u64 = 512 * 1024;
const REGION_LINES: usize = 70;
const REGION_CHARS: usize = 6_000;
const MAX_IDENTIFIERS: usize = 6;
const MAX_CANDIDATES: usize = 8;

/// A piece of code the request points at.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CodeRegion {
    /// Path relative to the workspace, with forward slashes.
    pub(super) path: String,
    /// One-based inclusive line range.
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) symbol: String,
    pub(super) definition: bool,
    pub(super) is_test: bool,
    pub(super) score: f64,
    pub(super) text: String,
}

impl CodeRegion {
    pub(super) fn label(&self) -> String {
        format!(
            "{} lines {}-{} ({} of `{}`)",
            self.path,
            self.start,
            self.end,
            if self.definition { "definition" } else { "use" },
            self.symbol
        )
    }

    /// Numbered lines as the model reads them with `read_file`.
    pub(super) fn numbered(&self) -> String {
        self.text
            .lines()
            .enumerate()
            .map(|(offset, line)| format!("{:>5}| {line}", self.start + offset))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn git(workspace: &Path, args: &[&str]) -> Option<String> {
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(workspace).args(args);
    crate::process_window::hide_std(&mut command);
    let output = command.output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|text| !text.is_empty())
}

/// The Python launcher that exists on this machine (`python3` on Linux and
/// macOS, `python` or `py -3` on Windows), so the model does not find out by
/// failing a command.
fn python_launcher() -> &'static str {
    static LAUNCHER: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    LAUNCHER.get_or_init(|| {
        let candidates: &[(&str, &[&str])] = if cfg!(windows) {
            &[("python", &[]), ("py", &["-3"])]
        } else {
            &[("python3", &[]), ("python", &[])]
        };
        for (program, args) in candidates {
            let mut command = std::process::Command::new(program);
            command.args(*args).arg("--version");
            crate::process_window::hide_std(&mut command);
            if command.output().is_ok_and(|output| output.status.success()) {
                return match (*program, args.is_empty()) {
                    ("py", false) => "py -3",
                    ("python3", _) => "python3",
                    _ => "python",
                };
            }
        }
        "python3"
    })
}

/// How tests run here, from the project files; `None` when nothing is recognised.
fn detect_test_command(workspace: &Path) -> Option<(&'static str, String)> {
    let has = |name: &str| workspace.join(name).exists();
    if has("package.json") {
        let scripts = std::fs::read_to_string(workspace.join("package.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|json| {
                json.get("scripts")
                    .and_then(|scripts| scripts.get("test"))
                    .cloned()
            });
        if scripts.is_some() {
            let runner = if has("yarn.lock") {
                "yarn test"
            } else if has("pnpm-lock.yaml") {
                "pnpm test"
            } else {
                "npm test"
            };
            return Some(("Node", runner.into()));
        }
    }
    if has("Cargo.toml") {
        return Some(("Rust", "cargo test".into()));
    }
    if has("go.mod") {
        return Some(("Go", "go test ./...".into()));
    }
    if has("pom.xml") {
        return Some(("Java (Maven)", "mvn test".into()));
    }
    if has("build.gradle") || has("build.gradle.kts") {
        return Some(("Java (Gradle)", "gradle test".into()));
    }
    let python = has("pyproject.toml")
        || has("setup.py")
        || has("setup.cfg")
        || has("tox.ini")
        || has("pytest.ini")
        || has("requirements.txt");
    if python {
        let launcher = python_launcher();
        let install = if has("pyproject.toml") || has("setup.py") {
            " Setup when imports fail or pytest is missing: `pip install -e . pytest` (pip installs go into an isolated environment that later commands use automatically)."
        } else {
            " If pytest is missing: `pip install pytest` (goes into an isolated environment)."
        };
        return Some(("Python", format!("{launcher} -m pytest -q.{install}")));
    }
    None
}

/// A short description of the workspace for the model; `None` when it does not
/// look like a project.
pub(super) fn workspace_brief(workspace: &Path) -> Option<String> {
    let test = detect_test_command(workspace);
    let branch = git(workspace, &["rev-parse", "--abbrev-ref", "HEAD"]);
    if test.is_none() && branch.is_none() {
        return None;
    }
    let mut lines = vec![format!(
        "Workspace: {} (commands start here; the shell keeps its directory and exported variables between calls)",
        workspace.display()
    )];
    if let Some(branch) = branch {
        let status = git(workspace, &["status", "--short"]).unwrap_or_default();
        let changed = status.lines().count();
        lines.push(format!(
            "Git: branch {branch}, {}",
            if changed == 0 {
                "clean".to_owned()
            } else {
                format!(
                    "{changed} uncommitted change(s):\n{}",
                    status.lines().take(12).collect::<Vec<_>>().join("\n")
                )
            }
        ));
        if let Some(log) = git(workspace, &["log", "--oneline", "-3"]) {
            lines.push(format!("Recent commits:\n{log}"));
        }
    }
    let mut entries = std::fs::read_dir(workspace)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.')
                || matches!(
                    name.as_str(),
                    "node_modules" | "target" | "__pycache__" | "dist" | "build" | "venv"
                )
            {
                return None;
            }
            let dir = entry.file_type().ok()?.is_dir();
            Some(if dir { format!("{name}/") } else { name })
        })
        .collect::<Vec<_>>();
    entries.sort();
    if !entries.is_empty() {
        let shown = entries
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("  ");
        lines.push(format!(
            "Top level: {shown}{}",
            if entries.len() > 40 { "  …" } else { "" }
        ));
    }
    if let Some((stack, command)) = test {
        lines.push(format!("Project type: {stack}. Run tests with: {command}"));
    }
    Some(lines.join("\n"))
}

/// Words in the request that look like code: `name()`, `snake_case`,
/// `camelCase`, or anything in backticks.
fn identifiers_in(request: &str) -> Vec<String> {
    const COMMON: &[&str] = &[
        "this",
        "that",
        "with",
        "from",
        "have",
        "should",
        "returns",
        "return",
        "function",
        "method",
        "class",
        "file",
        "files",
        "test",
        "tests",
        "bug",
        "report",
        "repository",
        "library",
        "source",
        "number",
        "numbers",
        "value",
        "values",
        "string",
        "example",
        "false",
        "true",
        "none",
        "null",
    ];
    let mut found: Vec<String> = Vec::new();
    let mut push = |word: &str| {
        let word =
            word.trim_matches(|character: char| !(character.is_alphanumeric() || character == '_'));
        let lowered = word.to_ascii_lowercase();
        if word.len() >= 4
            && word
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
            && !COMMON.contains(&lowered.as_str())
            && !found.iter().any(|existing| existing == word)
        {
            found.push(word.to_owned());
        }
    };
    let backticked = Regex::new(r"`([^`\n]{1,80})`").expect("valid regex");
    for capture in backticked.captures_iter(request) {
        for part in capture[1].split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            push(part);
        }
    }
    let word = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").expect("valid regex");
    for matched in word.find_iter(request) {
        let text = matched.as_str();
        let followed_by_call = request[matched.end()..].starts_with('(');
        let snake = text.contains('_') && !text.starts_with('_');
        let camel =
            text.chars().skip(1).any(char::is_uppercase) && text.chars().any(char::is_lowercase);
        if followed_by_call || snake || camel {
            push(text);
        }
    }
    found.truncate(MAX_IDENTIFIERS);
    found
}

fn is_test_path(path: &str) -> bool {
    let lowered = path.to_ascii_lowercase();
    lowered
        .split('/')
        .any(|part| matches!(part, "test" | "tests" | "spec" | "specs" | "__tests__"))
        || lowered.rsplit('/').next().is_some_and(|name| {
            name.starts_with("test_")
                || name.contains("_test.")
                || name.contains(".test.")
                || name.contains(".spec.")
        })
}

/// Code regions the request points at: definitions of the identifiers it
/// names (best), uses of them in tests, and files it names by path.
pub(super) fn code_candidates(workspace: &Path, request: &str) -> Vec<CodeRegion> {
    let identifiers = identifiers_in(request);
    let named_files = Regex::new(r"[\w./\\-]+\.[A-Za-z]{1,5}")
        .expect("valid regex")
        .find_iter(request)
        .map(|found| found.as_str().replace('\\', "/"))
        .filter(|path| path.contains('/') && workspace.join(path).is_file())
        .collect::<Vec<_>>();
    if identifiers.is_empty() && named_files.is_empty() {
        return Vec::new();
    }
    let definition_patterns = identifiers
        .iter()
        .map(|name| {
            let escaped = regex::escape(name);
            (
                name.clone(),
                Regex::new(&format!(
                    r"^\s*(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+|export\s+|static\s+|public\s+|private\s+|protected\s+)*(?:def|fn|function|class|struct|enum|trait|interface|type|const|func|impl|module|object)\s+{escaped}\b|^\s*(?:export\s+)?(?:const|let|var)\s+{escaped}\s*[=:]|^{escaped}\s*[=:]"
                ))
                .expect("valid regex"),
                Regex::new(&format!(r"\b{escaped}\b")).expect("valid regex"),
            )
        })
        .collect::<Vec<_>>();
    let mut regions: Vec<CodeRegion> = Vec::new();
    let walker = ignore::WalkBuilder::new(workspace)
        .hidden(true)
        .git_ignore(true)
        .build();
    for entry in walker
        .filter_map(|entry| entry.ok())
        .take(MAX_FILES_SCANNED)
    {
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let relative = entry
            .path()
            .strip_prefix(workspace)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace('\\', "/");
        let test = is_test_path(&relative);
        let lines = text.lines().collect::<Vec<_>>();
        let window = |from: usize, count: usize| {
            let start = from.saturating_sub(1).max(0);
            let end = (start + count).min(lines.len());
            let mut body = lines[start..end].join("\n");
            if body.len() > REGION_CHARS {
                body = body.chars().take(REGION_CHARS).collect();
            }
            (start + 1, end, body)
        };
        for (name, definition, mention) in &definition_patterns {
            let mut mentioned = false;
            for (index, line) in lines.iter().enumerate() {
                if definition.is_match(line) {
                    let (start, end, body) = window(index.saturating_sub(1) + 1, REGION_LINES);
                    regions.push(CodeRegion {
                        path: relative.clone(),
                        start,
                        end,
                        symbol: name.clone(),
                        definition: true,
                        is_test: test,
                        score: if test { 0.6 } else { 1.0 },
                        text: body,
                    });
                    break;
                }
                if !mentioned && test && mention.is_match(line) {
                    mentioned = true;
                    let (start, end, body) = window(index.saturating_sub(10) + 1, 40);
                    regions.push(CodeRegion {
                        path: relative.clone(),
                        start,
                        end,
                        symbol: name.clone(),
                        definition: false,
                        is_test: true,
                        score: 0.5,
                        text: body,
                    });
                }
            }
        }
        if named_files.contains(&relative) && !regions.iter().any(|region| region.path == relative)
        {
            let (start, end, body) = window(1, 100);
            regions.push(CodeRegion {
                path: relative.clone(),
                start,
                end,
                symbol: relative.clone(),
                definition: false,
                is_test: test,
                score: 0.8,
                text: body,
            });
        }
    }
    regions.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.start.cmp(&right.start))
    });
    regions.dedup_by(|left, right| left.path == right.path && left.start == right.start);
    regions.truncate(MAX_CANDIDATES);
    regions
}

/// The message that hands the chosen regions to the model.
pub(super) fn prefetched_code_block(regions: &[CodeRegion]) -> String {
    let body = regions
        .iter()
        .map(|region| format!("--- {}\n{}", region.label(), region.numbered()))
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "<prefetched_code>\nThe request names these code locations; their current contents are below (line numbers as read_file shows them), so do not search for or read them again. Use read_file for anything beyond them.\n{body}\n</prefetched_code>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/pkg")).unwrap();
        std::fs::create_dir_all(dir.path().join("tests")).unwrap();
        std::fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\nname = \"pkg\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/pkg/numbers.py"),
            "import math\n\n\ndef unrelated():\n    return 1\n\n\ndef carry_over(value, fmt=\"%.1f\"):\n    \"\"\"Round and carry.\"\"\"\n    return math.floor(value)\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tests/test_numbers.py"),
            "from pkg import carry_over\n\n\ndef test_carry():\n    assert carry_over(1.5) == 1\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn identifiers_are_the_words_that_look_like_code() {
        let found = identifiers_in(
            "intword() returns 1000.0 for `big_value` and parseConfig fails; the function is slow in tests/test_x.py",
        );
        assert!(found.contains(&"intword".to_owned()), "{found:?}");
        assert!(found.contains(&"big_value".to_owned()));
        assert!(found.contains(&"parseConfig".to_owned()));
        assert!(!found.contains(&"function".to_owned()));
    }

    #[test]
    fn the_definition_and_its_test_are_found_from_the_request() {
        let dir = project();
        let regions = code_candidates(dir.path(), "carry_over() mishandles rounding up; fix it");
        assert_eq!(regions[0].path, "src/pkg/numbers.py");
        assert!(
            regions[0].definition && regions[0].start <= 8,
            "{:?}",
            regions[0]
        );
        assert!(regions[0].text.contains("def carry_over"));
        assert!(
            regions
                .iter()
                .any(|region| region.is_test && region.path == "tests/test_numbers.py")
        );
        let block = prefetched_code_block(&regions[..1]);
        assert!(
            block.contains("   8| def carry_over") || block.contains("def carry_over"),
            "{block}"
        );
        // Nothing code-like in the request: nothing is fetched.
        assert!(code_candidates(dir.path(), "please tidy things up").is_empty());
    }

    #[test]
    fn a_path_named_in_the_request_is_included() {
        let dir = project();
        let regions = code_candidates(
            dir.path(),
            "update tests/test_numbers.py to cover negatives",
        );
        assert!(
            regions
                .iter()
                .any(|region| region.path == "tests/test_numbers.py")
        );
    }

    #[test]
    fn the_brief_says_where_things_are_and_how_tests_run() {
        let dir = project();
        let brief = workspace_brief(dir.path()).unwrap();
        assert!(
            brief.contains("pyproject.toml") && brief.contains("src/") && brief.contains("tests/"),
            "{brief}"
        );
        assert!(brief.contains("-m pytest -q"), "{brief}");
        assert!(brief.contains("pip install -e . pytest"), "{brief}");
        let empty = tempfile::tempdir().unwrap();
        assert!(workspace_brief(empty.path()).is_none());
    }
}
