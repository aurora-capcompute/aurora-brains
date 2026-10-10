//! Web pages reduced to readable text.

use std::cell::RefCell;
use std::rc::Rc;

// html_to_text extracts readable text from HTML with lol_html (Cloudflare's
// streaming rewriter): it keeps only ordinary `Data` text — so <script>/<style>/
// <title> content, which the parser tags as ScriptData/RawText, is dropped — and
// inserts a newline at block boundaries, then decodes common entities and
// collapses whitespace. On a rewriter error the raw input is returned unchanged.
pub(crate) fn html_to_text(html: &str) -> String {
    use lol_html::html_content::TextType;
    use lol_html::{element, rewrite_str, text, RewriteStrSettings};

    let out = Rc::new(RefCell::new(String::new()));
    let o_block = out.clone();
    let o_text = out.clone();

    let settings = RewriteStrSettings {
        element_content_handlers: vec![
            // Block-level elements: separate their text with a newline.
            element!(
                "p, div, br, li, tr, hr, section, article, header, footer, main, aside, nav, h1, h2, h3, h4, h5, h6, ul, ol, table, blockquote, pre, figure",
                move |_el| {
                    o_block.borrow_mut().push('\n');
                    Ok(())
                }
            ),
            // Keep only ordinary text; script/style/raw-text chunks are dropped.
            text!("*", move |t| {
                if t.text_type() == TextType::Data {
                    o_text.borrow_mut().push_str(t.as_str());
                }
                Ok(())
            }),
        ],
        ..RewriteStrSettings::default()
    };

    if rewrite_str(html, settings).is_err() {
        return html.to_string();
    }
    let raw = out.borrow().clone();
    collapse_ws(&decode_entities(&raw))
}

// decode_entities expands the handful of HTML entities lol_html leaves in text.
// "&amp;" is expanded last so an already-decoded "&lt;" is not re-decoded.
fn decode_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

// collapse_ws squeezes runs of spaces/tabs to one space and runs of newlines to
// at most a blank line, so extracted text reads as paragraphs without the
// original markup's whitespace.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    let mut pending_newlines = 0u32;
    for ch in s.chars() {
        if ch == '\n' || ch == '\r' {
            pending_newlines += 1;
            continue;
        }
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_newlines > 0 {
            if !out.is_empty() {
                out.push_str(if pending_newlines >= 2 { "\n\n" } else { "\n" });
            }
        } else if pending_space && !out.is_empty() {
            out.push(' ');
        }
        pending_space = false;
        pending_newlines = 0;
        out.push(ch);
    }
    out.trim().to_string()
}
