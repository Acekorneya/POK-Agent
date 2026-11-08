//! Syntax diagnostics after edits: JSON and TOML are parsed in-process, other
//! languages use their own checker when it is installed. Bounded in time and
//! output; a missing checker means no report, never an error.

use super::*;

const CHECK_TIMEOUT: Duration = Duration::from_secs(8);
const MAX_ERRORS: usize = 5;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct Diagnostic {
    pub line: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct DiagnosticReport {
    pub checked_with: &'static str,
    pub errors: Vec<Diagnostic>,
}

impl DiagnosticReport {
    /// One line per error for a reminder to the model.
    pub(crate) fn summary(&self) -> String {
        self.errors
            .iter()
            .map(|error| match error.line {
                Some(line) => format!("line {line}: {}", error.message),
                None => error.message.clone(),
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Check one file. `None` when the language has no checker here or it is not
/// installed.
pub(crate) async fn check_file(path: &Path) -> Option<DiagnosticReport> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "json" => {
            let text = std::fs::read_to_string(path).ok()?;
            Some(DiagnosticReport {
                checked_with: "json parser",
                errors: serde_json::from_str::<Value>(&text)
                    .err()
                    .map(|error| {
                        vec![Diagnostic {
                            line: Some(error.line()),
                            message: error.to_string(),
                        }]
                    })
                    .unwrap_or_default(),
            })
        }
        "toml" => {
            let text = std::fs::read_to_string(path).ok()?;
            Some(DiagnosticReport {
                checked_with: "toml parser",
                errors: text
                    .parse::<toml::Table>()
                    .err()
                    .map(|error| {
                        let line = error.span().map(|span| {
                            text[..span.start.min(text.len())].matches('\n').count() + 1
                        });
                        vec![Diagnostic {
                            line,
                            message: error.message().to_owned(),
                        }]
                    })
                    .unwrap_or_default(),
            })
        }
        "py" | "pyw" => python_syntax(path).await,
        "js" | "mjs" | "cjs" => {
            let output = run_checker("node", &["--check", &path.display().to_string()]).await?;
            Some(report("node --check", node_errors(&output)))
        }
        "sh" | "bash" if cfg!(unix) => {
            let output = run_checker("bash", &["-n", &path.display().to_string()]).await?;
            Some(report("bash -n", shell_errors(&output)))
        }
        "rs" => {
            let output = run_checker(
                "rustfmt",
                &["--check", "--edition", "2021", &path.display().to_string()],
            )
            .await?;
            Some(report("rustfmt", rust_errors(&output)))
        }
        _ => None,
    }
}

fn report(checked_with: &'static str, errors: Vec<Diagnostic>) -> DiagnosticReport {
    DiagnosticReport {
        checked_with,
        errors,
    }
}

/// The report as it goes into a tool result: omitted when nothing could check.
pub(crate) fn report_value(report: Option<&DiagnosticReport>) -> Value {
    match report {
        None => Value::Null,
        Some(report) if report.errors.is_empty() => {
            json!({"checked_with": report.checked_with, "syntax": "ok"})
        }
        Some(report) => json!({
            "checked_with": report.checked_with,
            "syntax": "errors",
            "errors": report.errors,
            "note": "the file has syntax errors; fix them before moving on",
        }),
    }
}

struct CheckerOutput {
    success: bool,
    stdout: String,
    stderr: String,
}

async fn run_checker(program: &str, args: &[&str]) -> Option<CheckerOutput> {
    let mut command = tokio::process::Command::new(program);
    command
        .args(args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    crate::process_window::hide_tokio(&mut command);
    let output = tokio::time::timeout(CHECK_TIMEOUT, command.output())
        .await
        .ok()?
        .ok()?;
    Some(CheckerOutput {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

const PYTHON_CHECK: &str = "import ast,sys\nsrc=open(sys.argv[1],encoding='utf-8').read()\ntry:\n    ast.parse(src,sys.argv[1])\nexcept SyntaxError as e:\n    print('%s|%s'%(e.lineno,e.msg))\n    sys.exit(1)\n";

async fn python_syntax(path: &Path) -> Option<DiagnosticReport> {
    let file = path.display().to_string();
    let candidates: &[(&str, &[&str])] = if cfg!(windows) {
        &[("python", &[]), ("py", &["-3"])]
    } else {
        &[("python3", &[]), ("python", &[])]
    };
    for (program, prefix) in candidates {
        let mut args = prefix.to_vec();
        args.extend(["-c", PYTHON_CHECK, file.as_str()]);
        let Some(output) = run_checker(program, &args).await else {
            continue;
        };
        if output.success {
            return Some(report("python ast", Vec::new()));
        }
        let errors = output
            .stdout
            .lines()
            .filter_map(|line| {
                let (number, message) = line.split_once('|')?;
                Some(Diagnostic {
                    line: number.trim().parse().ok(),
                    message: message.trim().to_owned(),
                })
            })
            .collect::<Vec<_>>();
        if errors.is_empty() {
            // The interpreter itself failed (not installed as expected): try the next.
            continue;
        }
        return Some(report("python ast", errors));
    }
    None
}

/// `file.js:12` then the offending line, then `SyntaxError: message`.
fn node_errors(output: &CheckerOutput) -> Vec<Diagnostic> {
    if output.success {
        return Vec::new();
    }
    let line = output.stderr.lines().find_map(|line| {
        line.rsplit_once(':')
            .and_then(|(_, n)| n.trim().parse().ok())
    });
    output
        .stderr
        .lines()
        .find(|line| line.contains("Error:"))
        .map(|message| {
            vec![Diagnostic {
                line,
                message: message.trim().to_owned(),
            }]
        })
        .unwrap_or_default()
}

/// `file.sh: line 3: syntax error near ...`
fn shell_errors(output: &CheckerOutput) -> Vec<Diagnostic> {
    if output.success {
        return Vec::new();
    }
    output
        .stderr
        .lines()
        .take(MAX_ERRORS)
        .map(|line| {
            let line_number = line
                .split("line ")
                .nth(1)
                .and_then(|rest| rest.split(':').next())
                .and_then(|number| number.trim().parse().ok());
            Diagnostic {
                line: line_number,
                message: line.rsplit(": ").next().unwrap_or(line).trim().to_owned(),
            }
        })
        .collect()
}

/// rustfmt reports a formatting difference as a diff (not an error) and a
/// parse failure as `error: ...` with `--> file:line:col`.
fn rust_errors(output: &CheckerOutput) -> Vec<Diagnostic> {
    let text = format!("{}\n{}", output.stderr, output.stdout);
    let lines = text.lines().collect::<Vec<_>>();
    let mut errors = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if let Some(message) = line.strip_prefix("error: ") {
            if message.starts_with("couldn't") || message.starts_with("could not") {
                continue;
            }
            let at = lines
                .get(index + 1..index + 4)
                .into_iter()
                .flatten()
                .find_map(|next| {
                    let located = next.trim().strip_prefix("--> ")?;
                    located.rsplit(':').nth(1)?.parse().ok()
                });
            errors.push(Diagnostic {
                line: at,
                message: message.to_owned(),
            });
        }
        if errors.len() >= MAX_ERRORS {
            break;
        }
    }
    errors
}

/// Errors still present in files the run changed, for the completion check.
pub(crate) async fn syntax_errors_in(paths: &[PathBuf]) -> Vec<(PathBuf, DiagnosticReport)> {
    let mut found = Vec::new();
    for path in paths {
        if let Some(report) = check_file(path).await
            && !report.errors.is_empty()
        {
            found.push((path.clone(), report));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn json_and_toml_errors_report_their_line() {
        let dir = tempfile::tempdir().unwrap();
        let bad_json = dir.path().join("a.json");
        std::fs::write(&bad_json, "{\n  \"a\": 1,\n  \"b\": \n}\n").unwrap();
        let report = check_file(&bad_json).await.unwrap();
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].line, Some(4));
        let good = dir.path().join("b.json");
        std::fs::write(&good, "{\"a\": 1}").unwrap();
        assert!(check_file(&good).await.unwrap().errors.is_empty());
        let bad_toml = dir.path().join("c.toml");
        std::fs::write(&bad_toml, "a = 1\nb = = 2\n").unwrap();
        let report = check_file(&bad_toml).await.unwrap();
        assert_eq!(report.errors[0].line, Some(2), "{report:?}");
        // No checker for an unknown extension.
        let other = dir.path().join("d.unknownext");
        std::fs::write(&other, "x").unwrap();
        assert!(check_file(&other).await.is_none());
    }

    #[tokio::test]
    async fn python_syntax_errors_are_caught_with_their_line() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("broken.py");
        // The escaped-quote mistake a model made in a live run.
        std::fs::write(&file, "def f():\n    return 1\nassert f() == \\\"1\\\"\n").unwrap();
        let Some(report) = check_file(&file).await else {
            eprintln!("skipped: no python on this machine");
            return;
        };
        assert_eq!(report.checked_with, "python ast");
        assert_eq!(report.errors[0].line, Some(3), "{report:?}");
        std::fs::write(&file, "def f():\n    return 1\n").unwrap();
        assert!(check_file(&file).await.unwrap().errors.is_empty());
        let summary = DiagnosticReport {
            checked_with: "x",
            errors: vec![Diagnostic {
                line: Some(3),
                message: "bad".into(),
            }],
        }
        .summary();
        assert_eq!(summary, "line 3: bad");
    }
}
