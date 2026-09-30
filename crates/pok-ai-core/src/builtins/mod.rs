use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::PathBuf,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use base64::Engine;
use chrono::{Local, Utc};
use regex::Regex;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    PokError, Result,
    grounding::build_targets,
    memory::{MemoryWriteProvenance, NewProcedure, ProcedureKind, ProcedureStep},
    policy::RiskClass,
    tool::{
        ActiveTaskState, DerivedObservationView, FocusedControl, InputLedger, PendingSubmission,
        PendingVisualLocalization, TaskItem, TaskItemStatus, Tool, ToolContext, ToolRegistry,
        UserQuestion, UserQuestionOption, UserQuestionOutcome, VerifiedSubmission,
    },
    types::{
        CaptureRequest, CaptureScope, CaptureTarget, DesktopCapture, InputAction,
        InteractionTarget, MonitorInfo, MouseButton, Observation, Rect, Screenshot, TargetSource,
        WindowInfo,
    },
};

mod action_batch;
mod browser_navigation;
mod capture;
mod fast_actions_tool;
mod input;
mod memory_tools;
mod planning;
mod scrolling;
mod targeting;
mod windows;

use action_batch::*;
use browser_navigation::*;
pub(crate) use capture::safe_filename;
use capture::*;
use fast_actions_tool::*;
use input::*;
use memory_tools::*;
use planning::*;
use scrolling::*;
use targeting::*;
use windows::*;

pub(crate) use action_batch::normalized_batch_kind;
pub(crate) use capture::model_observation_value;
pub use fast_actions_tool::{
    FastActionsArgs, FastBranch, FastInputStep, FastInterrupt, FastLeaf, FastLeafBranch,
    FastOperation, FastSubgoal,
};
pub(crate) use input::observation_id_for;

pub fn register_desktop_tools(registry: &mut ToolRegistry) {
    registry.register(CurrentTimeTool);
    registry.register(UpdateTaskPlanTool);
    registry.register(DiscoverToolsTool);
    registry.register(FastActionsTool);
    registry.register(RememberFactTool);
    registry.register(AskUserQuestionTool);
    registry.register(ObserveDesktopTool);
    registry.register(CaptureScreenTool);
    registry.register(InspectScreenRegionTool);
    registry.register(ListWindowsTool);
    registry.register(OpenApplicationTool);
    registry.register(ActivateWindowTool);
    registry.register(BrowserNavigateTool);
    registry.register(QueryScreenTextTool);
    registry.register(ReadClipboardTool);
    registry.register(QueryWindowTreeTool);
    registry.register(ClickTargetTool);
    registry.register(LocateVisualTargetTool);
    registry.register(ClickLocalizedTool);
    registry.register(MovePointerTool);
    registry.register(DragPointerTool);
    registry.register(DragTargetTool);
    registry.register(HoverTargetTool);
    registry.register(ScrollViewTool);
    registry.register(ScrollUntilTextTool);
    registry.register(TypeTextTool);
    registry.register(SimulateInputTool);
    registry.register(ExecuteActionBatchTool);
    crate::browser::register_managed_browser_tools(registry);
}

pub fn register_memory_tools(registry: &mut ToolRegistry) {
    registry.register(MemorySearchTool);
    registry.register(MemorySaveTool);
    registry.register(SkillSearchTool);
    registry.register(SkillLoadTool);
    registry.register(SkillCreateTool);
    registry.register(SessionContextSearchTool);
    registry.register(SessionContextReadTool);
}

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schema_for!(T)).expect("schema serializes")
}

const fn default_true() -> bool {
    true
}

#[cfg(test)]
pub(crate) mod tests;
