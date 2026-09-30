"""Write diagnostics/router_question_suite.json: hand-labelled router questions.

States come from real bench captures (scripts/build_router_suite.py) of the
Windows Settings app, File Explorer and the Rust Book in the managed browser,
reduced to the bounded shape production sends and stripped of personal
labels (account names, machine names). Discord is excluded because its
sidebar holds other people's names.

Kinds, matching what the delegated fast_actions loop asks:
  completion        is a planner done_when already true? (quoted and unquoted)
  target            which candidate matches the planner's goal and hint?
  condition_choice  which planner-written branch/interrupt condition holds?

Run `pok-ai router-bench diagnostics/router_question_suite.json` against each
backend; see docs/decision-router-backends.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "diagnostics" / "router_question_suite.json"


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^0-9a-z]+", " ", text.lower()).split() if len(w) >= 2]


def bucket(query: str, label: str) -> str:
    q, l = set(words(query)), words(label)
    if not q or not l:
        return "weakly task relevant"
    score = len([w for w in l if w in q]) / min(len(q), len(l))
    if score >= 0.8:
        return "strongly task relevant"
    if score >= 0.45:
        return "task relevant"
    return "weakly task relevant"


def desktop(title: str, app: str, labels: list[str]) -> dict:
    return {"window": {"application": app, "title": title}, "visible": labels[:24]}


def page(url: str, title: str, text: str) -> dict:
    return {"page": {"url": url, "title": title, "text_start": text[:200]}}


SETTINGS_HOME = desktop("Settings", "ApplicationFrameHost.exe", [
    "Home (list item, selected)", "System (list item)", "Bluetooth & devices (list item)",
    "Network & internet (list item)", "Personalization (list item)", "Apps (list item)",
    "Accounts (list item)", "Time & language (list item)", "Gaming (list item)",
    "Accessibility (list item)", "Privacy & security (list item)", "Windows Update (list item)",
    "Recommended settings (text)", "Display (list item)", "Installed apps (list item)",
    "Lock screen (list item)", "Windows Update Attention needed (button)", "Ethernet 2 Connected (button)",
    "Bluetooth devices (text)", "Add device (button)", "View all devices (button)",
    "Personalize your device (text)", "Windows (light), 1 images (list item)", "Glow, 4 images (list item)",
])
SETTINGS_DISPLAY = desktop("Settings", "ApplicationFrameHost.exe", [
    "System (breadcrumb bar item)", "Display (breadcrumb bar item)", "Home (list item)",
    "System (list item, selected)", "Bluetooth & devices (list item)", "Personalization (list item)",
    "Multiple displays (group)", "Identify (button)", "Brightness & color (text)",
    "Night light (toggle switch)", "Color profile (text)", "HDR (text)", "Scale & layout (text)",
    "Scale (combo box)", "100% (Recommended) (list item)", "Display resolution (combo box)",
    "2560 × 1440 (Recommended) (list item)", "Show more settings (button)", "Apps (list item)",
    "Accounts (list item)", "Gaming (list item)", "Accessibility (list item)",
    "Privacy & security (list item)", "Windows Update (list item)",
])
SETTINGS_ADVANCED = desktop("Settings", "ApplicationFrameHost.exe", [
    "System (breadcrumb bar item)", "Display (breadcrumb bar item)",
    "Advanced display (breadcrumb bar item)", "Advanced display (text)",
    "Select a display to view or change its settings (combo box)", "Display information (group)",
    "Desktop mode (text)", "2560 × 1440, 165 Hz (text)", "Active signal mode (text)",
    "Variable refresh rate (text)", "Not Supported (text)", "Bit depth (text)", "8-bit (text)",
    "Color format (text)", "RGB (text)", "Refresh rate (group)", "Refresh rate (combo box)",
    "Choose a refresh rate (text)", "165 Hz (list item)", "More about refresh rate (link)",
    "Dynamic refresh rate isn't supported (text)", "Home (list item)", "System (list item)",
    "Display adapter properties for Display 1 (link)",
])
SETTINGS_ERROR = desktop("Settings", "explorer.exe", [
    "Close (button)", "Settings (text)",
    "Windows cannot find 'Settings'. Make sure you typed the name correctly, and then try again. (text)",
    "OK (button)",
])
EXPLORER_THIS_PC = desktop("This PC - File Explorer", "explorer.exe", [
    "This PC (tab item)", "Address Bar (edit)", "Search This PC (edit)", "This PC (9 items) (group)",
    "Local Disk (C:) (list item)", "1TB SSD (D:) (list item)", "500GB_SSD (E:) (list item)",
    "GameDrive (F:) (list item)", "GameDriveHDD (G:) (list item)", "NZB (H:) (list item)",
    "Crucial_SSD_2 (I:) (list item)", "980 SSD (J:) (list item)", "USB (L:) (list item)",
    "Home (tree item)", "Gallery (tree item)", "OneDrive (tree item)", "Downloads (pinned) (tree item)",
    "Documents (pinned) (tree item)", "Pictures (pinned) (tree item)", "This PC (tree item)",
    "Network (tree item)", "Space used (edit)", "Available space (edit)", "Details (app bar button)",
])
EXPLORER_C = desktop("Local Disk (C:) - File Explorer", "explorer.exe", [
    "Local Disk (C:) (tab item)", "This PC (split button)", "Local Disk (C:) (split button)",
    "Search Local Disk (C:) (edit)", "$WinREAgent (list item)", "AMD (list item)", "Autodesk (list item)",
    "inetpub (list item)", "Intel (list item)", "MSI (list item)", "NVIDIA Corporation (list item)",
    "PerfLogs (list item)", "Program Files (list item)", "Program Files (x86) (list item)",
    "ProgramData (list item)", "tmp (list item)", "Users (list item)", "Windows (list item)",
    "XboxGames (list item)", "Home (tree item)", "This PC (tree item)", "Local Disk (C:) (tree item)",
    "Name (split button)", "Date modified (split button)",
])
EXPLORER_PROGRAM_FILES = desktop("Program Files - File Explorer", "explorer.exe", [
    "Program Files (tab item)", "This PC (split button)", "Local Disk (C:) (split button)",
    "Program Files (split button)", "Program Files (78 items) (group)", "Adobe (list item)",
    "Amazon (list item)", "AMD (list item)", "Application Verifier (list item)", "Autodesk (list item)",
    "AutoHotkey (list item)", "Bambu Studio (list item)", "Blender Foundation (list item)",
    "cFosSpeed (list item)", "Common Files (list item)", "Docker (list item)", "dotnet (list item)",
    "Git (list item)", "HWiNFO64 (list item)", "Intel (list item)", "Kodi (list item)", "LLVM (list item)",
    "Up to \"Local Disk (C:)\" (Alt + Up Arrow) (app bar button)", "Search Program Files (edit)",
])
BOOK_TEXT = "The Rust Programming Language Foreword Introduction 1. Getting Started 1.1. Installation 1.2. Hello, World! 1.3. Hello, Cargo! 2. Programming a Guessing Game 3. Common Programming Concepts"
BOOK_INDEX = page("https://doc.rust-lang.org/book/", "The Rust Programming Language - The Rust Programming Language", BOOK_TEXT)
BOOK_CH4 = page("https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html", "Understanding Ownership - The Rust Programming Language", BOOK_TEXT)
BOOK_4_1 = page("https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html", "What is Ownership? - The Rust Programming Language", BOOK_TEXT)
BOOK_LINKS = [
    "The Rust Programming Language", "Foreword", "Introduction", "1. Getting Started", "2. Programming a Guessing Game",
    "3. Common Programming Concepts", "3.5. Control Flow", "4. Understanding Ownership", "4.1. What is Ownership?",
    "4.2. References and Borrowing", "4.3. The Slice Type", "5. Using Structs to Structure Related Data",
    "6. Enums and Pattern Matching", "7. Packages, Crates, and Modules", "8. Common Collections", "9. Error Handling",
    "9.2. Recoverable Errors with Result", "10. Generic Types, Traits, and Lifetimes", "11. Writing Automated Tests",
    "13. Functional Language Features: Iterators and Closures",
]


def labels_of(state: dict, role: str) -> list[str]:
    suffix = f" ({role})"
    return [label[: -len(suffix)] for label in state.get("visible", []) if label.endswith(suffix)]


def desktop_candidates(state: dict, query: str, roles=("list item", "tree item", "link", "tab item")) -> list[dict]:
    candidates = []
    for label in state["visible"]:
        match = re.match(r"^(.*) \((list item|tree item|link|tab item)(, selected)?\)$", label)
        if not match or match.group(2) not in roles:
            continue
        name, role = match.group(1), match.group(2)
        candidates.append({
            "id": f"click_{len(candidates)}",
            "tool": "click_target",
            "arguments": {"observation_id": "obs_bench", "target_id": str(len(candidates)), "expected_label": name, "button": "left"},
            "description": f"{json.dumps(name)} ({role}, enabled, {bucket(query, name)})",
            "kind": "action",
            "local_score": 0.0,
        })
    return candidates


def browser_candidates(links: list[str]) -> list[dict]:
    return [{
        "id": f"browser_click_b{index}",
        "tool": "managed_browser_click",
        "arguments": {"snapshot_id": "bench", "target_id": f"b{index}"},
        "description": f"{json.dumps(name)} (browser link, visible)",
        "kind": "action",
        "local_score": 0.0,
    } for index, name in enumerate(links)]


def by_label(candidates: list[dict], name: str) -> str:
    for candidate in candidates:
        if candidate["description"].startswith(json.dumps(name) + " "):
            return candidate["id"]
    raise KeyError(name)


cases: list[dict] = []


def completion(case_id, goal, condition, state, expected):
    cases.append({"kind": "completion", "id": case_id, "goal": goal, "condition": condition, "state": state, "expected": expected})


def target(case_id, goal, hint, state, candidates, expected_label):
    cases.append({
        "kind": "target", "id": case_id, "goal": goal, "hint": hint, "candidates": candidates,
        "state": {key: value for key, value in state.items() if key in ("window", "page")},
        "expected": by_label(candidates, expected_label),
    })


def choice(case_id, goal, options, state, expected):
    cases.append({"kind": "condition_choice", "id": case_id, "goal": goal, "options": options, "state": state, "expected": expected})


NONE = ["none", "None of the other conditions is true right now"]

# Completion checks: unquoted (model judgment) and quoted (local grounding).
completion("done_home_not_system", "open System settings", "the System settings page is open", SETTINGS_HOME, "no")
completion("done_home_quoted_bluetooth", "open Bluetooth settings", '"Bluetooth & devices" is visible', SETTINGS_HOME, "yes")
completion("done_home_update_attention", "check Windows Update", "Windows Update needs attention", SETTINGS_HOME, "yes")
completion("done_display_open", "open Display settings", "the Display settings page is open", SETTINGS_DISPLAY, "yes")
completion("done_display_not_advanced", "open Advanced display", "the Advanced display page is open", SETTINGS_DISPLAY, "no")
completion("done_display_quoted_advanced", "open Advanced display", 'breadcrumb "Advanced display" is visible', SETTINGS_DISPLAY, "no")
completion("done_display_resolution", "read the resolution", "the display resolution is 2560 × 1440", SETTINGS_DISPLAY, "yes")
completion("done_advanced_open", "open Advanced display", "the Advanced display page is open", SETTINGS_ADVANCED, "yes")
completion("done_advanced_rate_shown", "read the refresh rate", "the refresh rate is shown", SETTINGS_ADVANCED, "yes")
completion("done_advanced_not_night_light", "open Night light", "the Night light settings page is open", SETTINGS_ADVANCED, "no")
completion("done_advanced_quoted_rate", "read the refresh rate", '"Refresh rate" and "165 Hz" are visible', SETTINGS_ADVANCED, "yes")
completion("done_error_dialog", "open Settings", "the Settings app is open", SETTINGS_ERROR, "no")
completion("done_thispc_not_program_files", "open Program Files", "the Program Files folder is open", EXPLORER_THIS_PC, "no")
completion("done_thispc_lists_drives", "open This PC", "This PC is open and lists drives", EXPLORER_THIS_PC, "yes")
completion("done_thispc_usb", "find a USB drive", "a USB drive is listed", EXPLORER_THIS_PC, "yes")
completion("done_c_open", "open the C: drive", "the Local Disk (C:) folder is open", EXPLORER_C, "yes")
completion("done_c_listed_not_open", "open Program Files", "the Program Files folder is open", EXPLORER_C, "no")
completion("done_c_not_windows", "open the Windows folder", "the Windows folder is open", EXPLORER_C, "no")
completion("done_pf_open", "open Program Files", "the Program Files folder is open", EXPLORER_PROGRAM_FILES, "yes")
completion("done_pf_quoted_adobe", "open Program Files", '"Adobe" is visible', EXPLORER_PROGRAM_FILES, "yes")
completion("done_pf_count", "open Program Files", "the folder shows 78 items", EXPLORER_PROGRAM_FILES, "yes")
completion("done_book_index_not_ch4", "open chapter 4", "the chapter 4 Understanding Ownership page is open", BOOK_INDEX, "no")
completion("done_book_ch4", "open chapter 4", "the chapter 4 Understanding Ownership page is open", BOOK_CH4, "yes")
completion("done_book_ch4_not_4_1", "open section 4.1", "the What is Ownership? section page is open", BOOK_CH4, "no")
completion("done_book_4_1", "open section 4.1", "the What is Ownership? section page is open", BOOK_4_1, "yes")
completion("done_book_quoted_url", "open chapter 4", 'the URL contains "ch04"', BOOK_INDEX, "no")
completion("done_book_not_search", "search the web", "a search results page is open", BOOK_INDEX, "no")

# Target picks: exact quoted hints (grounding should win), plain hints, and
# semantic goals with no shared words (the only place a model can add value).
home = desktop_candidates(SETTINGS_HOME, "")
target("pick_home_quoted_system", "open System settings", 'list item "System"', SETTINGS_HOME, home, "System")
target("pick_home_plain_system", "open System settings", "System in the left sidebar", SETTINGS_HOME, home, "System")
target("pick_home_semantic_wallpaper", "change the desktop background picture", "", SETTINGS_HOME, home, "Personalization")
target("pick_home_semantic_updates", "check for operating system updates", "", SETTINGS_HOME, home, "Windows Update")
target("pick_home_semantic_headphones", "pair new wireless headphones", "", SETTINGS_HOME, home, "Bluetooth & devices")
target("pick_home_semantic_language", "change the keyboard input language", "", SETTINGS_HOME, home, "Time & language")
thispc = desktop_candidates(EXPLORER_THIS_PC, "")
target("pick_thispc_c", "open the C: drive", "Local Disk (C:)", EXPLORER_THIS_PC, [c for c in thispc if "tree item" not in c["description"]], "Local Disk (C:)")
target("pick_thispc_semantic_usb", "open the removable USB stick", "", EXPLORER_THIS_PC, [c for c in thispc if "tree item" not in c["description"]], "USB (L:)")
cdrive = desktop_candidates(EXPLORER_C, "")
target("pick_c_program_files", "open the Program Files folder", '"Program Files"', EXPLORER_C, [c for c in cdrive if "tree item" not in c["description"]], "Program Files")
target("pick_c_semantic_32bit", "open the folder for 32-bit programs", "", EXPLORER_C, [c for c in cdrive if "tree item" not in c["description"]], "Program Files (x86)")
target("pick_c_semantic_profiles", "open the folder with user profiles", "", EXPLORER_C, [c for c in cdrive if "tree item" not in c["description"]], "Users")
pf = desktop_candidates(EXPLORER_PROGRAM_FILES, "")
target("pick_pf_git", "open the Git folder", "Git", EXPLORER_PROGRAM_FILES, [c for c in pf if "tree item" not in c["description"]], "Git")
target("pick_pf_semantic_blender", "open the folder of the Blender 3D app", "", EXPLORER_PROGRAM_FILES, [c for c in pf if "tree item" not in c["description"]], "Blender Foundation")
target("pick_pf_semantic_3dprint", "open the 3D printer slicer app folder", "", EXPLORER_PROGRAM_FILES, [c for c in pf if "tree item" not in c["description"]], "Bambu Studio")
book = browser_candidates(BOOK_LINKS)
target("pick_book_quoted_ch4", "open chapter 4", 'link "4. Understanding Ownership"', BOOK_INDEX, book, "4. Understanding Ownership")
target("pick_book_plain_ch4", "open chapter 4", "Understanding Ownership", BOOK_INDEX, book, "4. Understanding Ownership")
target("pick_book_semantic_errors", "open the chapter about handling errors", "", BOOK_INDEX, book, "9. Error Handling")
target("pick_book_semantic_tests", "open the chapter about writing tests", "", BOOK_INDEX, book, "11. Writing Automated Tests")
target("pick_book_semantic_borrowing", "learn about references and borrowing", "", BOOK_INDEX, book, "4.2. References and Borrowing")

# Condition choices: branches and interrupts written by the planner.
choice("branch_home_vs_error", "open Settings", [["c0", "the Settings home page is shown"], ["c1", "an error dialog is shown"], NONE], SETTINGS_HOME, "c0")
choice("branch_error_dialog", "open Settings", [["c0", "the Settings home page is shown"], ["c1", "an error dialog says the app could not be found"], NONE], SETTINGS_ERROR, "c1")
choice("branch_display_pages", "navigate Settings", [["c0", "the Sound page is open"], ["c1", "the Display page is open"], ["c2", "the Home page is open"], NONE], SETTINGS_DISPLAY, "c1")
choice("branch_advanced_vs_hdr", "navigate Settings", [["c0", "the HDR settings page is open"], ["c1", "the Advanced display page is open"], NONE], SETTINGS_ADVANCED, "c1")
choice("interrupt_home_signin_none", "open System", [["c0", "a sign-in prompt is shown"], NONE], SETTINGS_HOME, "none")
choice("branch_c_root_vs_pf", "open Program Files", [["c0", "the Program Files folder is open"], ["c1", "the C: drive root folder is open"], NONE], EXPLORER_C, "c1")
choice("branch_pf_contents", "list programs", [["c0", "the folder is empty"], ["c1", "the folder lists installed programs"], NONE], EXPLORER_PROGRAM_FILES, "c1")
choice("branch_thispc_quoted", "open the C: drive", [["c0", '"Local Disk (C:)" is visible'], ["c1", "the Program Files folder is open"], NONE], EXPLORER_THIS_PC, "c0")
choice("interrupt_book_cookie_none", "read chapter 4", [["c0", "a cookie consent banner is visible"], NONE], BOOK_CH4, "none")
choice("branch_book_toc", "open chapter 4", [["c0", "the book's table of contents is visible"], ["c1", "a login form is shown"], NONE], BOOK_INDEX, "c0")
choice("branch_book_chapter_vs_section", "open chapter 4", [["c0", "a chapter overview page is open"], ["c1", "the book title page is open"], NONE], BOOK_CH4, "c0")

OUT.write_text(json.dumps({
    "version": "1.0",
    "description": "POK-Agent router question suite: completion checks, target picks and condition choices from real captures",
    "cases": cases,
}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {len(cases)} cases to {OUT}")
