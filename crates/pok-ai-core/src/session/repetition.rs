//! Spotting a model that is stuck repeating itself, from the words it streams.
//!
//! A small model can fall into a loop in its own output: one word over and over
//! ("different different temperature different…"), or one sentence again and
//! again, until the token limit. The words it writes show it long before then.
//! [`RepetitionMonitor`] keeps the most recent words and measures, as
//! percentages, how much of them is the same few words and phrases. Ordinary
//! writing, lists, tables and code stay far below the limits; a loop is far
//! above them.

use std::collections::{HashMap, HashSet, VecDeque};

/// How many of the latest words are measured.
const WINDOW_WORDS: usize = 160;
/// Nothing is judged before this many words have been written.
const MIN_WORDS: usize = 120;
/// Words between two measurements, so streaming stays cheap.
const CHECK_EVERY: usize = 16;
/// Words in a phrase when counting repeated phrases.
const PHRASE_WORDS: usize = 4;
/// The share of phrases that already appeared earlier in the window.
const REPEATED_PHRASES_LIMIT: f64 = 0.70;
/// The share of the window taken by the single most common word. Ordinary
/// prose peaks near 7% (for "the").
const TOP_WORD_LIMIT: f64 = 0.40;
/// The least share of different words. Prose is above 40%.
const DISTINCT_WORDS_FLOOR: f64 = 0.10;

/// What was measured when the output was judged to be looping.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RepetitionReport {
    /// Words measured (the latest ones).
    pub(super) words: usize,
    /// The most common word and its share of those words (0 to 1).
    pub(super) top_word: String,
    pub(super) top_share: f64,
    /// The share of phrases that had already been written in this window.
    pub(super) repeated_phrases: f64,
    /// Different words as a share of all words.
    pub(super) distinct_words: f64,
    /// The end of what was written, to show what the loop looks like.
    pub(super) sample: String,
}

impl RepetitionReport {
    /// A sentence for the trace, the user and the model.
    pub(super) fn describe(&self) -> String {
        format!(
            "the word \"{}\" was {:.0}% of its last {} words, {:.0}% of its phrases had already been written, and only {:.0}% of the words were different",
            self.top_word,
            self.top_share * 100.0,
            self.words,
            self.repeated_phrases * 100.0,
            self.distinct_words * 100.0,
        )
    }
}

/// Reads streamed text a piece at a time.
#[derive(Debug, Default)]
pub(super) struct RepetitionMonitor {
    words: VecDeque<String>,
    /// The end of the text so far when it stops in the middle of a word.
    partial: String,
    total_words: usize,
    since_check: usize,
}

impl RepetitionMonitor {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Add the next piece of the stream. Returns a report the first time the
    /// recent words are mostly repeats.
    pub(super) fn push(&mut self, delta: &str) -> Option<RepetitionReport> {
        self.partial.push_str(delta);
        // Only whole words are counted: the text up to the last whitespace.
        let complete_end = self
            .partial
            .char_indices()
            .rev()
            .find(|(_, c)| c.is_whitespace())
            .map(|(index, c)| index + c.len_utf8())?;
        let complete: String = self.partial.drain(..complete_end).collect();
        let mut found = false;
        for token in complete.split_whitespace() {
            let Some(word) = normalize_word(token) else {
                continue;
            };
            if self.words.len() == WINDOW_WORDS {
                self.words.pop_front();
            }
            self.words.push_back(word);
            self.total_words += 1;
            self.since_check += 1;
            if self.since_check >= CHECK_EVERY {
                self.since_check = 0;
                found = true;
            }
        }
        if found && self.total_words >= MIN_WORDS {
            return self.measure();
        }
        None
    }

    fn measure(&self) -> Option<RepetitionReport> {
        let words: Vec<&str> = self.words.iter().map(String::as_str).collect();
        if words.len() < MIN_WORDS {
            return None;
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for word in &words {
            *counts.entry(word).or_default() += 1;
        }
        let (top_word, top_count) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(word, count)| ((*word).to_owned(), *count))?;
        let total = words.len() as f64;
        let top_share = top_count as f64 / total;
        let distinct_words = counts.len() as f64 / total;

        let mut seen: HashSet<&[&str]> = HashSet::new();
        let mut phrases = 0_usize;
        let mut repeated = 0_usize;
        for phrase in words.windows(PHRASE_WORDS) {
            phrases += 1;
            if !seen.insert(phrase) {
                repeated += 1;
            }
        }
        let repeated_phrases = if phrases == 0 {
            0.0
        } else {
            repeated as f64 / phrases as f64
        };

        let looping = repeated_phrases >= REPEATED_PHRASES_LIMIT
            || top_share >= TOP_WORD_LIMIT
            || distinct_words <= DISTINCT_WORDS_FLOOR;
        looping.then(|| RepetitionReport {
            words: words.len(),
            top_word,
            top_share,
            repeated_phrases,
            distinct_words,
            sample: words[words.len().saturating_sub(24)..].join(" "),
        })
    }
}

/// Whether the user asked for repeated output ("write hello 300 times", "say it
/// over and over"). Then repeating is the task, not a fault, and the guard
/// stays out of the way.
pub(super) fn request_asks_for_repetition(request: &str) -> bool {
    static ASKS: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    ASKS.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b(repeat\w*|over and over|again and again|on a loop|in a loop|\d[\d,]*\s*(x|times)|(hundred|thousand)s?\s+(of\s+)?times)\b",
        )
        .expect("a valid pattern")
    })
    .is_match(request)
}

/// A word without the punctuation around it, in lowercase. `None` for a token
/// with no letter or digit in it (a table rule, a bullet, an arrow).
fn normalize_word(token: &str) -> Option<String> {
    let trimmed = token.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `text` in small uneven pieces, as a stream would.
    fn stream(text: &str) -> Option<RepetitionReport> {
        let mut monitor = RepetitionMonitor::new();
        let mut rest = text;
        let mut size = 3;
        while !rest.is_empty() {
            let mut cut = size.min(rest.len());
            while !rest.is_char_boundary(cut) {
                cut += 1;
            }
            let (piece, tail) = rest.split_at(cut);
            if let Some(report) = monitor.push(piece) {
                return Some(report);
            }
            rest = tail;
            size = size % 11 + 2;
        }
        None
    }

    const PROSE: &str = "The harness reads the model's answer while it arrives. When the weather \
        report finished loading, the assistant compared three sources, noticed that two of them \
        disagreed about tomorrow's temperature, and decided to check the official forecast page \
        before answering. That page listed a high of eighty-one degrees with light wind from the \
        west, so the final summary mentions both numbers and explains where each one came from. \
        Later, the user asked about the weekend; the assistant opened a calendar, found two events \
        on Saturday afternoon, and suggested moving the picnic to Sunday morning when the air is \
        cooler. It also offered to set a reminder, wrote the note in a short sentence, and asked \
        whether the guest list should be shared with the neighbours who had helped last year.";

    #[test]
    fn ordinary_writing_is_never_flagged() {
        // The same paragraph several times over would repeat; distinct prose does not.
        assert_eq!(stream(PROSE), None);
        let long = format!(
            "{PROSE} Meanwhile the garden needed water, the car needed an oil change, and the \
             roof, which had leaked twice during the autumn storms, needed a proper inspection \
             before the next season arrived with its unpredictable rain and wind."
        );
        assert_eq!(stream(&long), None);
    }

    #[test]
    fn a_table_and_a_list_with_repeated_structure_are_not_a_loop() {
        let mut table = String::from("| Item | Price | Stock |\n| --- | --- | --- |\n");
        for index in 0..60 {
            table.push_str(&format!(
                "| Laptop model {index} | ${} | in stock |\n",
                200 + index * 7
            ));
        }
        assert_eq!(stream(&table), None, "table rows differ in their numbers");
        let mut list = String::new();
        for index in 0..50 {
            list.push_str(&format!(
                "- Result number {index} was found on page {} of the report\n",
                index * 3 + 1
            ));
        }
        assert_eq!(stream(&list), None);
    }

    /// The end of a reported run: one word, again and again.
    #[test]
    fn one_word_over_and_over_is_found_with_its_share() {
        let looped = "different. temperature. different - different different temperature.. \
            different different different temperature different different "
            .repeat(20);
        let report = stream(&looped).expect("a loop");
        assert_eq!(report.top_word, "different");
        assert!(report.top_share >= 0.40, "{report:?}");
        assert!(report.words >= MIN_WORDS);
        let sentence = report.describe();
        assert!(
            sentence.contains("the word \"different\" was"),
            "{sentence}"
        );
        assert!(sentence.contains('%'), "{sentence}");
    }

    #[test]
    fn a_sentence_repeated_again_and_again_is_found_by_its_phrases() {
        let looped =
            "I will now check the plan and then update the plan again so that the plan is current. "
                .repeat(14);
        let report = stream(&looped).expect("a loop");
        assert!(report.repeated_phrases >= 0.70, "{report:?}");
        assert!(report.sample.contains("plan"), "{report:?}");
    }

    #[test]
    fn a_loop_that_starts_after_normal_writing_is_still_caught() {
        let text = format!(
            "{PROSE} {}",
            "so the answer is the answer is the answer. ".repeat(30)
        );
        let report = stream(&text).expect("a loop");
        assert!(report.repeated_phrases >= 0.70 || report.top_share >= 0.40);
    }

    #[test]
    fn short_text_and_words_split_across_pieces_are_handled() {
        let mut monitor = RepetitionMonitor::new();
        // Too little to judge, however repetitive.
        assert_eq!(monitor.push("again again again again again "), None);
        // A word cut in two by the stream is one word.
        let mut monitor = RepetitionMonitor::new();
        monitor.push("hel");
        monitor.push("lo wor");
        monitor.push("ld ");
        assert_eq!(monitor.total_words, 2);
        // Punctuation-only tokens are not words.
        assert_eq!(normalize_word("---"), None);
        assert_eq!(normalize_word("**Done.**").as_deref(), Some("done"));
    }

    #[test]
    fn asking_for_repetition_switches_the_guard_off() {
        for asked in [
            "Write the word hello 300 times",
            "say it 50x",
            "Repeat this sentence ten times: hi",
            "keep saying it over and over",
            "print it a thousand times",
            "Count to 500 times 2",
        ] {
            assert!(request_asks_for_repetition(asked), "{asked}");
        }
        for ordinary in [
            "How is the weather outside",
            "Find the best laptop for $500",
            "Sometimes the page loads slowly; open it and summarise it",
            "Make notes on my desktop for these pages",
        ] {
            assert!(!request_asks_for_repetition(ordinary), "{ordinary}");
        }
    }

    #[test]
    fn non_ascii_text_does_not_break_the_cut() {
        // Accents, a non-breaking space and CJK, cut in odd places, without a panic.
        let text: String = (0..40)
            .map(|index| {
                format!(
                    "Hola José número {index}, más rápido que la computadora de {} años.\u{a0}日本語の文章 {index} です。 ",
                    index * 7 + 3
                )
            })
            .collect();
        assert_eq!(stream(&text), None);
    }
}
