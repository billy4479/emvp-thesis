// Standard alphanumeric (BibTeX `alpha`-style) citations and bibliography.
//
// Labels follow the classic alpha rules:
//   1 author   -> first three letters of the surname  [Har14]
//   2 authors  -> initials of both surnames           [NL18]
//   3-4 auth.  -> initials of every author           [KVPF20]
//   5+ authors -> initials of the first three + "+"   [BCH+25]
// followed by the last two digits of the year; colliding labels get the
// suffixes "a", "b", ...
//
// Typst's built-in CSL `citation-label` cannot reproduce these rules (it
// truncates the first surname for 4+ authors), so the entries are read
// directly from the Hayagriva YAML file and rendered natively.
//
// Entry point: `init-bib`, which returns the `cite` and `bibliography`
// functions bound to the given entries.

#let _authors-of(entry) = {
  let a = entry.at("author", default: "")
  if type(a) == array { a } else if a == "" or a == none { () } else { (a,) }
}

// "Sutherland, Andrew V." -> "Sutherland"; organizations pass through.
#let _family-name(raw) = {
  let raw = raw.trim()
  if "," in raw { raw.split(",").first().trim() } else { raw }
}

// "Sutherland, Andrew V." -> "Andrew V. Sutherland".
#let _display-name(raw) = {
  let raw = raw.trim()
  if "," in raw {
    let parts = raw.split(",")
    let family = parts.first().trim()
    let given = parts.slice(1).join(",").trim()
    if given == "" { family } else { given + " " + family }
  } else {
    raw
  }
}

// Last two digits of the publication year ("2018-06" -> "18").
#let _year(entry) = {
  let m = str(entry.at("date", default: "")).match(regex("\\d{4}"))
  if m == none { "" } else { m.text.slice(2) }
}

#let _full-year(entry) = {
  let d = str(entry.at("date", default: ""))
  let m = d.match(regex("\\d{4}"))
  if m == none { d } else { m.text }
}

#let _tag(entry) = {
  let surnames = _authors-of(entry).map(_family-name)
  if surnames.len() == 0 {
    "Unknown"
  } else if surnames.len() == 1 {
    let s = surnames.first()
    s.slice(0, calc.min(3, s.len()))
  } else if surnames.len() <= 4 {
    surnames.map(s => s.slice(0, 1)).join()
  } else {
    surnames.slice(0, 3).map(s => s.slice(0, 1)).join() + "+"
  }
}

// Map of citation key -> label, with "a", "b", ... disambiguation.
#let _labels(entries) = {
  let sorted = entries
    .pairs()
    .map(((k, e)) => (k, _tag(e) + _year(e)))
    .sorted(key: p => lower(p.at(1)) + " " + p.at(0))
  let out = (:)
  let prev = none
  let n = 0
  for p in sorted {
    let k = p.at(0)
    let lbl = p.at(1)
    if lbl == prev {
      n += 1
      out.insert(k, lbl + str.from-unicode(96 + n))
    } else {
      prev = lbl
      n = 0
      out.insert(k, lbl)
    }
  }
  out
}

#let _pages(s) = {
  if s == none { none } else { str(s).split("-").join("–") }
}

#let _vol-issue(parent) = {
  let vol = parent.at("volume", default: none)
  let issue = parent.at("issue", default: none)
  if vol == none { none } else {
    str(vol) + if issue == none { "" } else { "(" + str(issue) + ")" }
  }
}

// DOI / arXiv / bare URL links, rendered after the year.
// The serial number (ePrint number, RFC, article number) is rendered in the
// entry body instead, see `_entry-body`.
#let _suffix(entry) = {
  let parts = ()
  let url = entry.at("url", default: none)
  let sn = entry.at("serial-number", default: none)
  if type(sn) == dictionary {
    let doi = sn.at("doi", default: none)
    let arxiv = sn.at("arxiv", default: none)
    if doi != none { parts.push(link("https://doi.org/" + doi, [doi: #doi])) }
    if arxiv != none {
      let target = if url == none { "https://arxiv.org/abs/" + arxiv } else {
        url
      }
      parts.push(link(target, [arXiv: #arxiv]))
    }
  }
  if parts.len() == 0 and url != none { parts.push(link(url, [#url])) }
  if parts.len() == 0 { none } else { parts.join(", ") }
}

// The serial number, linked when a URL is available, or none.
#let _serial(entry) = {
  let sn = entry.at("serial-number", default: none)
  let serial = if type(sn) == dictionary {
    sn.at("serial", default: none)
  } else if type(sn) == str {
    sn
  } else {
    none
  }
  if serial == none { none } else { serial }
}

#let _entry-body(entry, et-al) = {
  let names = _authors-of(entry).map(_display-name)
  // A truncated list ends with italic "et al" (the ". " segment joiner
  // supplies the dot after it).
  let authors = if names.len() == 0 {
    none
  } else if names.len() > et-al {
    names.slice(0, et-al).join(", ") + [, #emph[et al]]
  } else if names.len() == 1 {
    names.first()
  } else if names.len() == 2 {
    names.first() + " and " + names.last()
  } else {
    names.slice(0, -1).join(", ") + ", and " + names.last()
  }

  let typ = entry.at("type", default: "Misc")
  let quoted = typ in ("Article", "Misc", "Chapter")

  // Container bits; the serial number and year are appended commonly below.
  let tail = ()
  let parent = entry.at("parent", default: none)
  if parent != none {
    let ptype = parent.at("type", default: "")
    let ptitle = emph(parent.at("title", default: ""))
    if ptype == "Periodical" {
      tail.push(ptitle)
      let vi = _vol-issue(parent)
      let end = _pages(entry.at("page-range", default: none))
      if end == none {
        // Without a page range the serial is the article number, rendered
        // as volume(issue):serial (e.g. ACM "Art. no. 3").
        let sn = entry.at("serial-number", default: (:))
        let art = if type(sn) == dictionary {
          sn.at("serial", default: none)
        } else { none }
        if art != none { end = str(art) }
      }
      if vi != none {
        tail.push(str(vi) + if end == none { "" } else { ":" + end })
      } else if end != none { tail.push(end) }
    } else {
      // Proceedings, Conference, Book, ...
      tail.push([In #ptitle])
      if parent.at("volume", default: none) != none {
        tail.push(str(parent.volume))
      }
      if entry.at("page-range", default: none) != none {
        tail.push([pages #_pages(entry.page-range)])
      }
    }
  } else if typ == "Report" {
    if entry.at("publisher", default: none) != none {
      tail.push(entry.publisher)
    }
  } else if entry.at("genre", default: none) != none {
    tail.push(entry.genre)
    if entry.at("organization", default: none) != none {
      tail.push(entry.organization)
    }
  }
  let serial = _serial(entry)
  // In journals without a page range the serial was consumed as the article
  // number above.
  let consumed-as-artno = (
    parent != none
      and parent.at("type", default: "") == "Periodical"
      and entry.at("page-range", default: none) == none
  )
  if serial != none and not consumed-as-artno { tail.push(serial) }
  if _full-year(entry) != "" { tail.push(_full-year(entry)) }

  tail = tail.filter(part => part != none and part != "")
  let rest = tail.join(", ")
  let suf = _suffix(entry)

  // Segments after the title are comma-joined and closed by one period
  // ("... Journal, 60:113-119, 2014, doi: ..."). The title carries its own
  // punctuation inside the quotes ("... “Title.” Journal, ..."); a title that
  // already ends in ., ! or ? is not given a second terminator.
  let dot = if entry.title.match(regex("[.!?]$")) == none { [.] } else { [] }
  let segs = ()
  if authors != none { segs.push([#authors. ]) }
  segs.push(if quoted { [“#entry.title#dot” ] } else { [#entry.title#dot ] })
  if rest != "" or suf != none {
    let bits = ()
    if rest != "" { bits.push(rest) }
    if suf != none { bits.push(suf) }
    segs.push(bits.join(", ") + [.])
  }
  segs.join("")
}

// Bind `cite` and `bibliography` to the given bibliography entries.
//
// The entries are loaded by the caller, so the yaml path is resolved
// relative to the caller's own file, with no hidden coupling to lib/:
//
//   #import "lib/alpha-bib.typ": init-bib
//   #let (cite, bibliography) = init-bib(yaml("BIBLIOGRAPHY.yml"))
//
//   #cite(<harvey2014>)   // [Har14], links to the bibliography entry
//   #bibliography(title: "Bibliography")
//
// With more than `et-al` authors, the bibliography lists the first `et-al`
// names followed by "et al."; pass a larger value or `none` to list everyone.
//
// Like Typst's built-in bibliography, only cited entries are shown unless
// `full: true`. Labels are computed once over all entries, so `cite` and
// `bibliography` always agree, also for collision suffixes.
#let init-bib(entries, et-al: 6) = {
  let labels = _labels(entries)

  let cite(..keys) = {
    let names = ()
    for k in keys.pos() {
      if str(k) not in labels { panic("no bibliography entry for key", str(k)) }
      names.push(str(k))
    }
    let rendered = (
      "[" + keys.pos().map(k => link(k, labels.at(str(k)))).join(", ") + "]"
    )
    [#rendered#metadata(names) <alpha-bib-cited>]
  }

  let bibliography(title: "Bibliography", full: false) = context {
    let shown = if full { entries.pairs() } else {
      let cited = query(<alpha-bib-cited>).map(m => m.value).flatten()
      entries.pairs().filter(p => p.at(0) in cited)
    }
    let sorted = shown.sorted(key: p => (
      lower(labels.at(p.at(0))) + " " + p.at(0)
    ))
    heading(numbering: none, title)
    v(0.65em)
    grid(
      columns: (auto, 1fr),
      column-gutter: 0.9em,
      row-gutter: 1.5em,
      ..sorted
        .map(p => {
          let k = p.at(0)
          ([#labels.at(k)#label(k)], _entry-body(p.at(1), et-al))
        })
        .flatten(),
    )
  }

  (cite: cite, bibliography: bibliography)
}
