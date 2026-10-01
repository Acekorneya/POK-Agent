//! The `fast_actions` tool: the planner's plan-tree arguments (steps, branches, interrupts, reads) executed by the session's fast-action runner.

use super::*;

/// Reversible operations a delegated `fast_actions` run may perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FastOperation {
    Click,
    /// Double-click a target, to open folders or files that a single click
    /// only selects.
    DoubleClick,
    /// Right-click a target to open its context menu.
    RightClick,
    /// Rest the pointer on a target, for menus that open on hover and tooltips.
    Hover,
    /// Drag the first quoted label in target_hint onto the second, e.g.
    /// "drag \"Sheet2\" onto \"Sheet1\"".
    Drag,
    Scroll,
    ActivateWindow,
    BrowserClick,
    BrowserScroll,
}

impl FastOperation {
    pub const ALL: [Self; 9] = [
        Self::Click,
        Self::DoubleClick,
        Self::RightClick,
        Self::Hover,
        Self::Drag,
        Self::Scroll,
        Self::ActivateWindow,
        Self::BrowserClick,
        Self::BrowserScroll,
    ];

    /// The harness tool a candidate must use to belong to this operation.
    pub fn tool(self) -> &'static str {
        match self {
            Self::Click | Self::DoubleClick | Self::RightClick => "click_target",
            Self::Hover => "hover_target",
            Self::Drag => "drag_target",
            Self::Scroll => "scroll_view",
            Self::ActivateWindow => "activate_window",
            Self::BrowserClick => "managed_browser_click",
            Self::BrowserScroll => "managed_browser_scroll",
        }
    }
}

/// One input step System 1 performs for you: press `key` (a named key or
/// plus-separated shortcut such as "Ctrl+Shift+F5") or type `text`. Give
/// exactly one of them.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct FastInputStep {
    #[serde(default)]
    #[schemars(length(max = 60))]
    pub key: Option<String>,
    #[serde(default)]
    #[schemars(length(max = 4000))]
    pub text: Option<String>,
    /// Replace the field's current content instead of adding to it.
    #[serde(default)]
    pub replace_existing: bool,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FastActionsArgs {
    /// The concrete subgoal for this run, e.g. "open the #general channel".
    #[schemars(length(min = 1, max = 500))]
    pub goal: String,
    /// The element to act on. Quote its exact visible label, e.g.
    /// "list item \"System\"", so it can be matched without a model.
    #[serde(default)]
    #[schemars(length(max = 300))]
    pub target_hint: Option<String>,
    /// Observable condition that means the subgoal is complete. Quote exact
    /// visible text, e.g. "heading \"Advanced display\" is visible", and it is
    /// checked locally; unquoted conditions are judged by the fast model.
    #[schemars(length(min = 1, max = 300))]
    pub done_when: String,
    /// Operations the run may use. Defaults to click and browser_click only;
    /// list scroll or activate_window explicitly when the subgoal needs them.
    #[serde(default)]
    pub allowed_operations: Vec<FastOperation>,
    /// Maximum actions per subgoal before returning to you (1-20, default 6).
    #[serde(default)]
    pub max_steps: Option<u32>,
    /// Phrases that must never be clicked anywhere in this plan, e.g.
    /// "voice channel". A candidate whose label contains one is skipped.
    #[serde(default)]
    pub avoid: Vec<String>,
    /// Rules checked before every step, for popups or dialogs that may
    /// appear. At most 4; at most 3 interrupt actions run per call.
    #[serde(default)]
    pub on_interrupt: Vec<FastInterrupt>,
    /// Follow-up subgoals run in order after this one completes. The run
    /// stops at the first subgoal that does not reach its done_when.
    #[serde(default)]
    pub then: Vec<FastSubgoal>,
    /// Alternatives chosen after the subgoal and every `then` step complete:
    /// the branch whose `when` holds runs next. Use "otherwise" for a default.
    #[serde(default)]
    pub branches: Vec<FastBranch>,
    /// Values to read after the plan finishes, each next to a quoted visible
    /// label, e.g. {"name": "refresh_rate", "label": "\"Refresh rate\""}.
    /// When every value is found the result omits the screenshot.
    #[serde(default)]
    pub read: Vec<crate::decision::ReadRequest>,
    /// Keys and text for System 1 to enter for this step, in order. With a
    /// target_hint, that field (found by its exact visible label) is focused
    /// first; without one, input goes to the focused control. Write every key
    /// and character yourself; System 1 only performs them and checks done_when.
    #[serde(default)]
    pub input: Vec<FastInputStep>,
}

/// A popup or dialog rule: when `when` holds, click `target_hint` (quote its
/// exact label) or, with `stop`, return to you.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FastInterrupt {
    #[schemars(length(min = 1, max = 300))]
    pub when: String,
    #[serde(default)]
    #[schemars(length(max = 300))]
    pub target_hint: Option<String>,
    #[serde(default)]
    pub stop: bool,
}

/// One plan node. Its own `branches` may hold one more level of plain steps.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FastSubgoal {
    #[schemars(length(min = 1, max = 500))]
    pub goal: String,
    #[serde(default)]
    #[schemars(length(max = 300))]
    pub target_hint: Option<String>,
    #[schemars(length(min = 1, max = 300))]
    pub done_when: String,
    /// Defaults to the parent call's allowed_operations.
    #[serde(default)]
    pub allowed_operations: Vec<FastOperation>,
    /// Extra phrases never to click in this node, added to the plan's avoid.
    #[serde(default)]
    pub avoid: Vec<String>,
    #[serde(default)]
    pub branches: Vec<FastLeafBranch>,
    /// Keys and text for System 1 to enter for this step (see the top level).
    #[serde(default)]
    pub input: Vec<FastInputStep>,
}

/// A branch whose steps may branch once more.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FastBranch {
    /// Condition for taking this branch; quote exact visible text when
    /// possible, or "otherwise" for the default branch.
    #[schemars(length(min = 1, max = 300))]
    pub when: String,
    pub then: Vec<FastSubgoal>,
}

/// A deepest-level branch of plain steps.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FastLeafBranch {
    #[schemars(length(min = 1, max = 300))]
    pub when: String,
    pub then: Vec<FastLeaf>,
}

/// A deepest-level plan step.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FastLeaf {
    #[schemars(length(min = 1, max = 500))]
    pub goal: String,
    #[serde(default)]
    #[schemars(length(max = 300))]
    pub target_hint: Option<String>,
    #[schemars(length(min = 1, max = 300))]
    pub done_when: String,
    #[serde(default)]
    pub allowed_operations: Vec<FastOperation>,
    #[serde(default)]
    pub avoid: Vec<String>,
    /// Keys and text for System 1 to enter for this step (see the top level).
    #[serde(default)]
    pub input: Vec<FastInputStep>,
}

/// Schema-only registration: `Session` intercepts this call and runs the
/// bounded fast-decision loop itself, so results stay one tool result.
pub(super) struct FastActionsTool;

#[async_trait]
impl Tool for FastActionsTool {
    fn name(&self) -> &'static str {
        "fast_actions"
    }

    fn description(&self) -> &'static str {
        "Run a plan of up to 64 steps, the whole path you already know (clicks, double-clicks to open folders or files, right-clicks, hovers, drags between two quoted labels, scrolls, window switches, managed-browser clicks, and keyboard input you write) with the fast decision model instead of issuing each action yourself. Quote exact visible labels in target_hint and done_when so they are checked locally. A step's input lists keys (\"Ctrl+Shift+F5\", \"Enter\") and text to enter in order, into the target_hint field or the focused control; you write every key and character, System 1 performs them and checks done_when. Chain steps with then, choose between alternatives with branches (\"otherwise\" as default), handle popups with on_interrupt, forbid targets with avoid, and read values with read. It never chooses text itself or commits consequential clicks. Returns status per step (done, unverified, uncertain, uncertain_branch, interrupted, popup (a dialog the plan did not expect, with its title, text, and buttons), stalled, no_candidates, budget_exhausted, unavailable), the executed steps (each with changed: what it visibly did), what_changed for the last action, read values, and the newest observation. Trust changed and what_changed instead of capturing again just to check that an action worked. Ambient notices (a tip, an update reminder, a rating prompt) with a Not now, Skip, Close, or similar button are put away automatically and listed in dismissed_popups; popups about data, money, access, or errors still come back as popup."
    }

    fn input_schema(&self) -> Value {
        schema::<FastActionsArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    async fn execute(&self, _args: Value, _context: &ToolContext) -> Result<Value> {
        Err(PokError::Tool(
            "fast_actions runs only inside an agent session with the delegated decision router"
                .into(),
        ))
    }
}
