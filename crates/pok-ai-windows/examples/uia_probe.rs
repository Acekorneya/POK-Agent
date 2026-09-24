//! Read-only UI Automation diagnostics. Performs no actions.
//!
//! Inspect named controls (type, supported patterns, legacy default action):
//!   `uia_probe "<window title part>" "<control name>" ...`
//!
//! Measure how much of a window the agent can operate without the mouse:
//!   `uia_probe --coverage "<window title part>" ...`
//! Coverage prints only control types and class names, never control names
//! or text, so it is safe to run on windows showing private content.

#[cfg(windows)]
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("--coverage") {
        for title in &args[1..] {
            coverage::report(title);
        }
    } else {
        named::report(&args);
    }
}

#[cfg(windows)]
fn window(automation: &uiautomation::UIAutomation, title: &str) -> Option<uiautomation::UIElement> {
    let walker = automation.get_control_view_walker().ok()?;
    let desktop = automation.get_root_element().ok()?;
    walker
        .get_children(&desktop)
        .unwrap_or_default()
        .into_iter()
        .find(|element| element.get_name().unwrap_or_default().contains(title))
}

#[cfg(windows)]
mod named {
    use uiautomation::{
        UIAutomation,
        patterns::{
            UIExpandCollapsePattern, UIInvokePattern, UILegacyIAccessiblePattern,
            UISelectionItemPattern, UITogglePattern,
        },
        types::{TreeScope, UIProperty},
    };

    pub fn report(args: &[String]) {
        let Some((title, names)) = args.split_first() else {
            eprintln!("usage: uia_probe <window title part> <control name>...");
            return;
        };
        let automation = UIAutomation::new().expect("UI Automation");
        let Some(window) = super::window(&automation, title) else {
            println!("no window containing {title:?}");
            return;
        };
        println!("window: {}", window.get_name().unwrap_or_default());
        for name in names {
            let condition = automation
                .create_property_condition(UIProperty::Name, name.as_str().into(), None)
                .expect("condition");
            let found = window
                .find_all(TreeScope::Descendants, &condition)
                .unwrap_or_default();
            if found.is_empty() {
                println!("{name:<22} (not found)");
            }
            for element in found {
                let mut patterns = Vec::new();
                if element.get_pattern::<UIInvokePattern>().is_ok() {
                    patterns.push("Invoke");
                }
                if element.get_pattern::<UISelectionItemPattern>().is_ok() {
                    patterns.push("SelectionItem");
                }
                if element.get_pattern::<UIExpandCollapsePattern>().is_ok() {
                    patterns.push("ExpandCollapse");
                }
                if element.get_pattern::<UITogglePattern>().is_ok() {
                    patterns.push("Toggle");
                }
                let default_action = element
                    .get_pattern::<UILegacyIAccessiblePattern>()
                    .and_then(|pattern| pattern.get_default_action())
                    .unwrap_or_default();
                println!(
                    "{name:<22} type={:<12} class={:<32} patterns={:<40} default_action={default_action:?}",
                    format!("{:?}", element.get_control_type().ok()),
                    element.get_classname().unwrap_or_default(),
                    patterns.join(","),
                );
            }
        }
    }
}

#[cfg(windows)]
mod coverage {
    use std::collections::BTreeMap;

    use uiautomation::{
        UIAutomation, UIElement,
        patterns::{
            UIExpandCollapsePattern, UIInvokePattern, UIRangeValuePattern, UISelectionItemPattern,
            UITextPattern, UITogglePattern, UIValuePattern,
        },
        types::{ControlType, TreeScope},
    };

    /// How the agent can operate one control.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Reach {
        /// Handled cursor-free by today's pattern rules.
        Now,
        /// Exposes a pattern the rules could use (menus, drop-downs, text
        /// values, sliders).
        Extension,
        /// A text surface that takes typing once focused (no mouse needed).
        Keyboard,
        /// Nothing usable is exposed: only a real mouse click works.
        MouseOnly,
    }

    fn classify(element: &UIElement, control: ControlType) -> Option<Reach> {
        let invoke = element.get_pattern::<UIInvokePattern>().is_ok();
        let select = element.get_pattern::<UISelectionItemPattern>().is_ok();
        let toggle = element.get_pattern::<UITogglePattern>().is_ok();
        let expand = element.get_pattern::<UIExpandCollapsePattern>().is_ok();
        let writable_value = element
            .get_pattern::<UIValuePattern>()
            .and_then(|pattern| pattern.is_readonly())
            .is_ok_and(|read_only| !read_only);
        let range = element.get_pattern::<UIRangeValuePattern>().is_ok();
        let text = element.get_pattern::<UITextPattern>().is_ok();
        Some(match control {
            ControlType::Button
            | ControlType::Hyperlink
            | ControlType::MenuItem
            | ControlType::SplitButton => {
                if invoke {
                    Reach::Now
                } else if expand || toggle {
                    Reach::Extension
                } else {
                    Reach::MouseOnly
                }
            }
            ControlType::CheckBox => {
                if toggle {
                    Reach::Now
                } else if invoke {
                    Reach::Extension
                } else {
                    Reach::MouseOnly
                }
            }
            ControlType::RadioButton | ControlType::TabItem => {
                if select {
                    Reach::Now
                } else if invoke {
                    Reach::Extension
                } else {
                    Reach::MouseOnly
                }
            }
            ControlType::ListItem | ControlType::TreeItem | ControlType::DataItem => {
                if select || invoke {
                    Reach::Now
                } else if expand || toggle {
                    Reach::Extension
                } else {
                    Reach::MouseOnly
                }
            }
            ControlType::ComboBox => {
                if expand || writable_value {
                    Reach::Extension
                } else {
                    Reach::MouseOnly
                }
            }
            ControlType::Edit | ControlType::Document => {
                if writable_value {
                    Reach::Extension
                } else if text || element.is_keyboard_focusable().unwrap_or(false) {
                    Reach::Keyboard
                } else {
                    Reach::MouseOnly
                }
            }
            ControlType::Slider | ControlType::Spinner => {
                if range {
                    Reach::Extension
                } else {
                    Reach::MouseOnly
                }
            }
            _ => return None,
        })
    }

    pub fn report(title: &str) {
        let automation = UIAutomation::new().expect("UI Automation");
        let Some(window) = super::window(&automation, title) else {
            println!("{title:<24} (no window)");
            return;
        };
        let condition = automation.create_true_condition().expect("condition");
        let elements = window
            .find_all(TreeScope::Descendants, &condition)
            .unwrap_or_default();
        let mut counts = BTreeMap::<Reach, usize>::new();
        let mut mouse_only = BTreeMap::<String, usize>::new();
        for element in elements.iter().take(5_000) {
            if !element.is_enabled().unwrap_or(false) {
                continue;
            }
            let Ok(control) = element.get_control_type() else {
                continue;
            };
            let Some(reach) = classify(element, control) else {
                continue;
            };
            *counts.entry(reach).or_default() += 1;
            if reach == Reach::MouseOnly {
                let key = format!(
                    "{control:?}/{}",
                    element.get_classname().unwrap_or_default()
                );
                *mouse_only.entry(key).or_default() += 1;
            }
        }
        let total = counts.values().sum::<usize>().max(1);
        let percent =
            |reach: Reach| 100.0 * *counts.get(&reach).unwrap_or(&0) as f64 / total as f64;
        let mut worst = mouse_only.into_iter().collect::<Vec<_>>();
        worst.sort_by(|left, right| right.1.cmp(&left.1));
        println!(
            "{title:<24} controls={total:<5} now={:>5.1}%  +extension={:>5.1}%  keyboard={:>5.1}%  mouse-only={:>5.1}%  top mouse-only: {}",
            percent(Reach::Now),
            percent(Reach::Now) + percent(Reach::Extension),
            percent(Reach::Keyboard),
            percent(Reach::MouseOnly),
            worst
                .iter()
                .take(3)
                .map(|(key, count)| format!("{key} x{count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("uia_probe requires Windows");
}
