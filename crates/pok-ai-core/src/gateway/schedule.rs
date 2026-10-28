//! Scheduled jobs: a prompt that runs on its own every few minutes or once a
//! day, with the answer sent to the home chat.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Local, NaiveTime, TimeZone, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::{PokError, Result};

/// Shorter intervals would let a typo flood the agent.
const MIN_INTERVAL_MINUTES: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Schedule {
    Every { minutes: u32 },
    Daily { hour: u8, minute: u8 },
}

impl Schedule {
    pub fn describe(&self) -> String {
        match self {
            Self::Every { minutes } if minutes % 60 == 0 => format!("every {} h", minutes / 60),
            Self::Every { minutes } => format!("every {minutes} min"),
            Self::Daily { hour, minute } => format!("daily at {hour:02}:{minute:02}"),
        }
    }

    /// The first moment after `base` this schedule fires, in local time.
    fn next_after(&self, base: DateTime<Local>) -> DateTime<Local> {
        match *self {
            Self::Every { minutes } => base + Duration::minutes(i64::from(minutes)),
            Self::Daily { hour, minute } => {
                let time = NaiveTime::from_hms_opt(u32::from(hour), u32::from(minute), 0)
                    .unwrap_or_default();
                let mut day = base.date_naive();
                loop {
                    // A time skipped by a clock change falls through to tomorrow.
                    if let Some(at) = Local.from_local_datetime(&day.and_time(time)).earliest()
                        && at > base
                    {
                        return at;
                    }
                    day = day.succ_opt().unwrap_or(day);
                }
            }
        }
    }
}

/// `every 30m`, `every 2 hours`, `daily 09:00`, `every day at 9:30`, followed
/// by the prompt. Returns the schedule and the rest of the text.
pub fn parse_schedule(text: &str) -> Option<(Schedule, String)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let lower: Vec<String> = words.iter().map(|word| word.to_lowercase()).collect();
    let (schedule, used) = match lower.first().map(String::as_str)? {
        "daily" => {
            let skip = usize::from(lower.get(1).map(String::as_str) == Some("at"));
            let (hour, minute) = parse_clock(lower.get(1 + skip)?)?;
            (Schedule::Daily { hour, minute }, 2 + skip)
        }
        "every" => {
            if lower.get(1).map(String::as_str) == Some("day") {
                let skip = usize::from(lower.get(2).map(String::as_str) == Some("at"));
                let (hour, minute) = parse_clock(lower.get(2 + skip)?)?;
                (Schedule::Daily { hour, minute }, 3 + skip)
            } else {
                let (amount, unit, used) = split_interval(&lower)?;
                let minutes = match unit.as_str() {
                    "m" | "min" | "mins" | "minute" | "minutes" => amount,
                    "h" | "hr" | "hrs" | "hour" | "hours" => amount.checked_mul(60)?,
                    _ => return None,
                };
                if minutes < MIN_INTERVAL_MINUTES {
                    return None;
                }
                (Schedule::Every { minutes }, used)
            }
        }
        _ => return None,
    };
    let prompt = words[used.min(words.len())..].join(" ");
    (!prompt.is_empty()).then_some((schedule, prompt))
}

/// `30m` or `30 minutes` after `every`: (amount, unit, words used in total).
fn split_interval(lower: &[String]) -> Option<(u32, String, usize)> {
    let first = lower.get(1)?;
    let digits: String = first.chars().take_while(char::is_ascii_digit).collect();
    let amount: u32 = digits.parse().ok()?;
    let attached = &first[digits.len()..];
    if attached.is_empty() {
        Some((amount, lower.get(2)?.clone(), 3))
    } else {
        Some((amount, attached.to_owned(), 2))
    }
}

fn parse_clock(word: &str) -> Option<(u8, u8)> {
    let (hour, minute) = word.split_once(':').unwrap_or((word, "0"));
    let hour: u8 = hour.parse().ok()?;
    let minute: u8 = minute.parse().ok()?;
    (hour < 24 && minute < 60).then_some((hour, minute))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub schedule: Schedule,
    pub enabled: bool,
    pub created: DateTime<Utc>,
    #[serde(default)]
    pub last_run: Option<DateTime<Utc>>,
}

impl Job {
    fn is_due(&self, now: DateTime<Local>) -> bool {
        let base = self.last_run.unwrap_or(self.created).with_timezone(&Local);
        self.enabled && self.schedule.next_after(base) <= now
    }
}

pub struct JobStore {
    path: PathBuf,
    jobs: Mutex<Vec<Job>>,
}

impl JobStore {
    pub fn open(path: &Path) -> Self {
        let jobs = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self {
            path: path.to_owned(),
            jobs: Mutex::new(jobs),
        }
    }

    fn save(&self, jobs: &[Job]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = self.path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(jobs)?)?;
        std::fs::rename(&temporary, &self.path)?;
        Ok(())
    }

    pub fn list(&self) -> Vec<Job> {
        self.jobs.lock().clone()
    }

    pub fn add(&self, name: &str, prompt: &str, schedule: Schedule) -> Result<Job> {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err(PokError::Config("a scheduled job needs a prompt".into()));
        }
        let name = name.trim();
        let job = Job {
            id: uuid::Uuid::new_v4().simple().to_string()[..6].to_owned(),
            name: if name.is_empty() {
                prompt.chars().take(40).collect()
            } else {
                name.to_owned()
            },
            prompt: prompt.to_owned(),
            schedule,
            enabled: true,
            created: Utc::now(),
            last_run: None,
        };
        let mut jobs = self.jobs.lock();
        jobs.push(job.clone());
        self.save(&jobs)?;
        Ok(job)
    }

    pub fn remove(&self, id: &str) -> Result<bool> {
        let mut jobs = self.jobs.lock();
        let before = jobs.len();
        jobs.retain(|job| job.id != id);
        let removed = jobs.len() != before;
        if removed {
            self.save(&jobs)?;
        }
        Ok(removed)
    }

    pub fn set_enabled(&self, id: &str, enabled: bool) -> Result<bool> {
        let mut jobs = self.jobs.lock();
        let Some(job) = jobs.iter_mut().find(|job| job.id == id) else {
            return Ok(false);
        };
        job.enabled = enabled;
        // Turning a job back on starts its clock from now.
        if enabled {
            job.last_run = Some(Utc::now());
        }
        self.save(&jobs)?;
        Ok(true)
    }

    /// Jobs whose time has come at `now`.
    pub fn due(&self, now: DateTime<Local>) -> Vec<Job> {
        self.jobs
            .lock()
            .iter()
            .filter(|job| job.is_due(now))
            .cloned()
            .collect()
    }

    pub fn mark_run(&self, id: &str, at: DateTime<Utc>) -> Result<()> {
        let mut jobs = self.jobs.lock();
        if let Some(job) = jobs.iter_mut().find(|job| job.id == id) {
            job.last_run = Some(at);
            self.save(&jobs)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_parse_in_the_forms_people_type() {
        let parsed = |text: &str| parse_schedule(text);
        assert_eq!(
            parsed("every 30m Check the build"),
            Some((Schedule::Every { minutes: 30 }, "Check the build".into()))
        );
        assert_eq!(
            parsed("every 2 hours summarize my notes"),
            Some((
                Schedule::Every { minutes: 120 },
                "summarize my notes".into()
            ))
        );
        assert_eq!(
            parsed("daily 09:00 morning briefing"),
            Some((
                Schedule::Daily { hour: 9, minute: 0 },
                "morning briefing".into()
            ))
        );
        assert_eq!(
            parsed("Every day at 7:45 water the plants"),
            Some((
                Schedule::Daily {
                    hour: 7,
                    minute: 45
                },
                "water the plants".into()
            ))
        );
        assert_eq!(
            parsed("daily at 18 wrap up"),
            Some((
                Schedule::Daily {
                    hour: 18,
                    minute: 0
                },
                "wrap up".into()
            ))
        );
    }

    #[test]
    fn bad_or_reckless_schedules_are_rejected() {
        assert!(
            parse_schedule("every 1m spam").is_none(),
            "under the minimum interval"
        );
        assert!(parse_schedule("daily 25:00 x").is_none());
        assert!(
            parse_schedule("daily 09:00").is_none(),
            "a job needs a prompt"
        );
        assert!(parse_schedule("weekly monday x").is_none());
        assert!(parse_schedule("every soon x").is_none());
        assert!(parse_schedule("").is_none());
    }

    fn at(hour: u32, minute: u32) -> DateTime<Local> {
        Local
            .with_ymd_and_hms(2026, 10, 28, hour, minute, 0)
            .earliest()
            .unwrap()
    }

    #[test]
    fn an_interval_job_is_due_once_its_interval_has_passed() {
        let job = Job {
            id: "a".into(),
            name: "n".into(),
            prompt: "p".into(),
            schedule: Schedule::Every { minutes: 30 },
            enabled: true,
            created: at(8, 0).with_timezone(&Utc),
            last_run: None,
        };
        assert!(!job.is_due(at(8, 29)));
        assert!(job.is_due(at(8, 30)));
        let ran = Job {
            last_run: Some(at(8, 30).with_timezone(&Utc)),
            ..job.clone()
        };
        assert!(!ran.is_due(at(8, 59)));
        assert!(ran.is_due(at(9, 0)));
        assert!(
            !Job {
                enabled: false,
                ..job
            }
            .is_due(at(12, 0))
        );
    }

    #[test]
    fn a_daily_job_fires_once_per_day_at_its_time() {
        let schedule = Schedule::Daily { hour: 9, minute: 0 };
        let created = Job {
            id: "a".into(),
            name: "n".into(),
            prompt: "p".into(),
            schedule,
            enabled: true,
            created: at(8, 0).with_timezone(&Utc),
            last_run: None,
        };
        assert!(!created.is_due(at(8, 59)));
        assert!(created.is_due(at(9, 0)));
        let ran = Job {
            last_run: Some(at(9, 0).with_timezone(&Utc)),
            ..created.clone()
        };
        assert!(!ran.is_due(at(23, 59)), "not again the same day");
        // Created after today's time: the first run is tomorrow, not now.
        let late = Job {
            created: at(10, 0).with_timezone(&Utc),
            ..created
        };
        assert!(!late.is_due(at(10, 1)));
    }

    #[test]
    fn jobs_persist_and_can_be_paused_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gateway").join("jobs.json");
        let store = JobStore::open(&path);
        let job = store
            .add(
                "",
                "Summarize the day",
                Schedule::Daily {
                    hour: 17,
                    minute: 30,
                },
            )
            .unwrap();
        assert_eq!(job.name, "Summarize the day");
        assert_eq!(job.schedule.describe(), "daily at 17:30");
        assert_eq!(Schedule::Every { minutes: 120 }.describe(), "every 2 h");
        assert!(
            store
                .add("x", "  ", Schedule::Every { minutes: 5 })
                .is_err()
        );

        let reopened = JobStore::open(&path);
        assert_eq!(reopened.list().len(), 1);
        assert!(reopened.set_enabled(&job.id, false).unwrap());
        assert!(!reopened.list()[0].enabled);
        reopened.mark_run(&job.id, Utc::now()).unwrap();
        assert!(reopened.list()[0].last_run.is_some());
        assert!(reopened.remove(&job.id).unwrap());
        assert!(!reopened.remove(&job.id).unwrap());
        assert!(JobStore::open(&path).list().is_empty());
    }
}
