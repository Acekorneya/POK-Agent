use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    PokError, Result,
    types::{CaptureScope, InputAction, Observation},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    ReadOnly,
    WorkspaceWrite,
    ProcessExecution,
    DesktopInput,
    HighImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum PolicyDecision {
    Allow,
    RequireApproval { reason: String },
    Deny { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyMode {
    Interactive,
    Autonomous,
    Exam {
        sandbox: PathBuf,
        allowed_window_title: String,
        allowed_process: String,
    },
}

#[derive(Debug, Clone)]
pub struct Policy {
    pub mode: PolicyMode,
}

impl Policy {
    pub fn interactive() -> Self {
        Self {
            mode: PolicyMode::Interactive,
        }
    }

    pub fn autonomous() -> Self {
        Self {
            mode: PolicyMode::Autonomous,
        }
    }

    pub fn decide(
        &self,
        tool: &str,
        risk: RiskClass,
        target_path: Option<&Path>,
    ) -> PolicyDecision {
        match (&self.mode, risk) {
            (_, RiskClass::ReadOnly) => PolicyDecision::Allow,
            (PolicyMode::Exam { .. }, RiskClass::HighImpact) => PolicyDecision::Deny {
                reason: "OS-critical actions are disabled in exam sandboxes".into(),
            },
            (_, RiskClass::HighImpact) => PolicyDecision::RequireApproval {
                reason: format!("{tool} can modify OS-critical or machine security state"),
            },
            (PolicyMode::Autonomous, _) => PolicyDecision::Allow,
            (_, _) if tool == "promote_helper_tool" => PolicyDecision::RequireApproval {
                reason: "promote_helper_tool installs a persistent reusable project tool".into(),
            },
            (PolicyMode::Interactive, RiskClass::DesktopInput) => PolicyDecision::Allow,
            (PolicyMode::Interactive, RiskClass::WorkspaceWrite | RiskClass::ProcessExecution) => {
                PolicyDecision::RequireApproval {
                    reason: format!("{tool} can change computer state"),
                }
            }
            (
                PolicyMode::Exam { sandbox, .. },
                RiskClass::WorkspaceWrite | RiskClass::ProcessExecution,
            ) => {
                if target_path.is_some_and(|path| path.starts_with(sandbox)) {
                    PolicyDecision::Allow
                } else {
                    PolicyDecision::Deny {
                        reason: "target is outside the exam sandbox".into(),
                    }
                }
            }
            (PolicyMode::Exam { .. }, RiskClass::DesktopInput) => PolicyDecision::Allow,
        }
    }

    pub fn validate_input(&self, action: &InputAction, observation: &Observation) -> Result<()> {
        let window =
            observation
                .foreground_window
                .as_ref()
                .ok_or_else(|| PokError::PolicyDenied {
                    tool: "simulate_input".into(),
                    reason: "no foreground window".into(),
                })?;
        if window.elevated {
            return Err(PokError::PolicyDenied {
                tool: "simulate_input".into(),
                reason: "elevated windows are blocked".into(),
            });
        }
        if observation
            .ui_elements
            .iter()
            .any(|element| element.password)
            && matches!(action, InputAction::TypeText { .. })
        {
            return Err(PokError::PolicyDenied {
                tool: "simulate_input".into(),
                reason: "typing while a password control is present is blocked".into(),
            });
        }
        if let PolicyMode::Exam {
            allowed_window_title,
            allowed_process,
            ..
        } = &self.mode
        {
            if !window.title.contains(allowed_window_title)
                || !window.process_name.eq_ignore_ascii_case(allowed_process)
            {
                return Err(PokError::PolicyDenied {
                    tool: "simulate_input".into(),
                    reason: "foreground window is outside exam allowlist".into(),
                });
            }
        }
        let points = match action {
            InputAction::Move { x, y }
            | InputAction::Click { x, y, .. }
            | InputAction::DoubleClick { x, y, .. } => vec![(*x, *y)],
            InputAction::Drag {
                start_x,
                start_y,
                end_x,
                end_y,
                ..
            } => vec![(*start_x, *start_y), (*end_x, *end_y)],
            _ => Vec::new(),
        };
        let pointer_bounds = match &self.mode {
            PolicyMode::Exam { .. } => &window.bounds,
            PolicyMode::Interactive | PolicyMode::Autonomous => observation
                .target
                .as_ref()
                .filter(|target| !matches!(target.scope, CaptureScope::All))
                .map_or(&window.bounds, |target| &target.bounds),
        };
        if points.iter().any(|(x, y)| !pointer_bounds.contains(*x, *y)) {
            return Err(PokError::PolicyDenied {
                tool: "simulate_input".into(),
                reason: "pointer action path is outside the authorized capture bounds".into(),
            });
        }
        Ok(())
    }
}

#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    async fn approve(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
        reason: &str,
    ) -> Result<ApprovalDecision>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    Deny,
    AllowOnce,
    AllowSession,
}

pub struct DenyApprovals;

#[async_trait]
impl ApprovalHandler for DenyApprovals {
    async fn approve(
        &self,
        _tool: &str,
        _arguments: &serde_json::Value,
        _reason: &str,
    ) -> Result<ApprovalDecision> {
        Ok(ApprovalDecision::Deny)
    }
}

pub struct AllowApprovals;

#[async_trait]
impl ApprovalHandler for AllowApprovals {
    async fn approve(
        &self,
        _tool: &str,
        _arguments: &serde_json::Value,
        _reason: &str,
    ) -> Result<ApprovalDecision> {
        Ok(ApprovalDecision::AllowOnce)
    }
}

pub fn deny_approvals() -> Arc<dyn ApprovalHandler> {
    Arc::new(DenyApprovals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    use crate::types::{CaptureTarget, InputAction, Observation, Rect, UiElement, WindowInfo};

    fn observation(title: &str, process: &str, password: bool) -> Observation {
        Observation {
            version: Uuid::new_v4(),
            captured_at: Utc::now(),
            cursor: None,
            screenshots: Vec::new(),
            ocr: Vec::new(),
            target: None,
            foreground_window: Some(WindowInfo {
                id: "1:test".into(),
                title: title.into(),
                process_name: process.into(),
                bounds: Rect {
                    x: 0,
                    y: 0,
                    width: 800,
                    height: 600,
                },
                elevated: false,
                visible: true,
                minimized: false,
            }),
            ui_elements: vec![UiElement {
                name: "field".into(),
                control_type: "edit".into(),
                automation_id: None,
                value: None,
                bounds: Rect {
                    x: 10,
                    y: 10,
                    width: 100,
                    height: 20,
                },
                enabled: true,
                password,
                offscreen: false,
                keyboard_focusable: true,
                clickable_point: None,
                selected: None,
                focused: false,
                desktop_shell: false,
            }],
            targets: Vec::new(),
            timings_ms: Default::default(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn exam_rejects_wrong_foreground_window() {
        let policy = Policy {
            mode: PolicyMode::Exam {
                sandbox: PathBuf::from("fixture"),
                allowed_window_title: "POK-Ai Brain Exam".into(),
                allowed_process: "msedge.exe".into(),
            },
        };
        assert!(
            policy
                .validate_input(
                    &InputAction::Click {
                        x: 20,
                        y: 20,
                        button: crate::types::MouseButton::Left
                    },
                    &observation("Notepad", "notepad.exe", false)
                )
                .is_err()
        );
    }

    #[test]
    fn typing_is_blocked_when_password_control_is_present() {
        assert!(
            Policy::interactive()
                .validate_input(
                    &InputAction::TypeText {
                        text: "hidden".into(),
                        replace_existing: false,
                    },
                    &observation("App", "app.exe", true)
                )
                .is_err()
        );
    }

    #[test]
    fn interactive_desktop_input_is_autonomous() {
        assert!(matches!(
            Policy::interactive().decide("simulate_input", RiskClass::DesktopInput, None),
            PolicyDecision::Allow
        ));
    }

    #[test]
    fn interactive_process_execution_still_requires_approval() {
        assert!(matches!(
            Policy::interactive().decide("run_command", RiskClass::ProcessExecution, None),
            PolicyDecision::RequireApproval { .. }
        ));
    }

    #[test]
    fn autonomous_only_prompts_for_os_critical_actions() {
        assert!(matches!(
            Policy::interactive().decide("promote_helper_tool", RiskClass::WorkspaceWrite, None),
            PolicyDecision::RequireApproval { .. }
        ));
        assert!(matches!(
            Policy::autonomous().decide("promote_helper_tool", RiskClass::WorkspaceWrite, None),
            PolicyDecision::Allow
        ));
        assert!(matches!(
            Policy::autonomous().decide("run_command", RiskClass::ProcessExecution, None),
            PolicyDecision::Allow
        ));
        assert!(matches!(
            Policy::autonomous().decide("run_command", RiskClass::HighImpact, None),
            PolicyDecision::RequireApproval { .. }
        ));
    }

    #[test]
    fn interactive_monitor_capture_allows_taskbar_outside_foreground_window() {
        let mut observation = observation("Browser", "browser.exe", false);
        observation.target = Some(CaptureTarget {
            scope: CaptureScope::Monitor,
            id: "primary".into(),
            title: "Primary monitor".into(),
            process_name: String::new(),
            bounds: Rect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            },
        });
        assert!(
            Policy::interactive()
                .validate_input(
                    &InputAction::Click {
                        x: 960,
                        y: 1060,
                        button: crate::types::MouseButton::Left,
                    },
                    &observation,
                )
                .is_ok()
        );
    }

    #[test]
    fn region_capture_authorizes_only_the_inspected_rectangle() {
        let mut observation = observation("Fixture", "fixture.exe", false);
        observation.target = Some(CaptureTarget {
            scope: CaptureScope::Region,
            id: "region:100:900:800:180".into(),
            title: "Screen region".into(),
            process_name: String::new(),
            bounds: Rect {
                x: 100,
                y: 900,
                width: 800,
                height: 180,
            },
        });
        let click = |x, y| InputAction::Click {
            x,
            y,
            button: crate::types::MouseButton::Left,
        };
        assert!(
            Policy::interactive()
                .validate_input(&click(500, 1_000), &observation)
                .is_ok()
        );
        assert!(
            Policy::interactive()
                .validate_input(&click(50, 1_000), &observation)
                .is_err()
        );
    }

    #[test]
    fn drag_requires_both_endpoints_inside_the_authorized_capture() {
        let mut observation = observation("Fixture", "fixture.exe", false);
        observation.target = Some(CaptureTarget {
            scope: CaptureScope::Region,
            id: "region".into(),
            title: "Region".into(),
            process_name: String::new(),
            bounds: Rect {
                x: 100,
                y: 100,
                width: 300,
                height: 300,
            },
        });
        let drag = |end_x| InputAction::Drag {
            start_x: 150,
            start_y: 150,
            end_x,
            end_y: 300,
            button: crate::types::MouseButton::Left,
            duration_ms: 350,
        };
        assert!(
            Policy::interactive()
                .validate_input(&drag(350), &observation)
                .is_ok()
        );
        assert!(
            Policy::interactive()
                .validate_input(&drag(450), &observation)
                .is_err()
        );
    }
}
