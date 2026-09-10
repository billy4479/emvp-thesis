// Standard alphanumeric (BibTeX `alpha`-style) citations and bibliography.
//
// Labels follow the classic alpha rules:
//   1 author   -> first three letters of the surname  [Har14]
//   2 authors  -> initials of both surnames           [NL18]
//   3-4 auth.  -> initials of every author           [KVPF20]
//   5+ authors -> initials of the first three + "+"   [BCH+25]
// followed by the last two digits of the year; colliding labels get the
// suffixes "a", "b", ... Entries without authors fall back to their editors,
// then to "Unknown".
//
// Typst's built-in CSL `citation-label` cannot reproduce these rules (it
// truncates the first surname for 4+ authors), so the entries are read
// directly from the Hayagriva YAML file and rendered natively. The renderer
// targets the full Hayagriva file format
// (https://github.com/typst/hayagriva/blob/main/docs/file-format.md):
//
//   - all entry types (case-insensitive), with default parent types;
//   - persons as strings ("Family[, Suffix[, Given]]", lowercase leading
//     words form the prefix) or {name, given-name, prefix, suffix, alias};
//   - single or multiple `parent`s, nested to any depth;
//   - formattable strings: plain, {value, verbatim, short}, and $math$;
//   - dates YYYY[-MM[-DD]] with negative years ("50 B.C.E.");
//   - fields: title, author, editor, date, parent, genre, organization,
//     location, publisher ({name, location} included), issue, volume,
//     chapter, edition, page-range, runtime, time-range, url (with access
//     date), serial-number (doi, arxiv, serial, isbn, issn, pmid, pmcid),
//     affiliated (with role phrases), archive, archive-location,
//     call-number.
//
// Not rendered (accepted and ignored, like in most CSL styles): `abstract`,
// `page-total`, `volume-total`, `language`. `note` is rendered with
// `show-notes: true`; this repository uses notes as internal annotations,
// so it defaults to false.
//
// Entry point: `init-bib`, which returns the `cite`, `citesec` and
// `bibliography` functions bound to the given entries. Tests live in
// `bibliography-tests.typ` ("typst compile handout/lib/bibliography-tests.typ").

// ---- formattable strings ------------------------------------------------

// A formattable string is a plain value or a mapping with a `value` subfield
// (plus optional `verbatim`/`short` metadata, which this style does not act
// on). This yields its plain text.
#let _fs-str(v) = {
  if v == none { "" } else if type(v) == dictionary {
    _fs-str(v.at("value", default: ""))
  } else { str(v) }
}

// A formattable string as content; balanced `$...$` spans are evaluated as
// Typst math, anything else is kept verbatim.
#let _fs-content(v) = {
  let s = _fs-str(v)
  let parts = s.split("$")
  if parts.len() >= 3 and calc.odd(parts.len()) {
    parts
      .enumerate()
      .map(((i, p)) => if calc.odd(i) {
        eval("$" + p + "$", mode: "markup")
      } else { p })
      .join("")
  } else {
    s
  }
}

// ---- persons ------------------------------------------------------------

// Array join that yields "" for empty arrays (Typst's `join` yields `none`).
#let _join(list, sep) = if list.len() == 0 { "" } else { list.join(sep) }

// A person is a Hayagriva string or a mapping with the subfields `name`,
// `given-name`, `prefix`, `suffix`, and `alias`. Returns
// (raw:, display:) where `raw` is the family part including the prefix
// (used for the alpha tag, e.g. "von zur Gathen" -> [vG13]) and `display`
// is the name in "Given Prefix Family, Suffix" order.
#let _person(p) = {
  let given = ""
  let suffix = ""
  let raw = ""
  if type(p) == dictionary {
    let pre = _fs-str(p.at("prefix", default: ""))
    let name = _fs-str(p.at("name", default: ""))
    raw = if pre == "" { name } else { pre + " " + name }
    given = _fs-str(p.at("given-name", default: ""))
    suffix = _fs-str(p.at("suffix", default: ""))
  } else {
    raw = str(p).trim()
    let parts = raw.split(",").map(s => s.trim())
    raw = parts.first()
    if parts.len() >= 3 {
      // "Family, Suffix, Given"
      suffix = parts.at(1)
      given = parts.slice(2).join(", ")
    } else if parts.len() == 2 {
      given = parts.at(1)
    }
  }
  // BibTeX rule: consecutive lowercase words at the start are the prefix.
  let words = raw.split(" ")
  let prefix = ()
  while (
    words.len() > 1
      and words.first() != ""
      and words.first() == lower(words.first())
  ) {
    prefix.push(words.remove(0))
  }
  let prefix-str = _join(prefix, " ")
  let family = words.join(" ")
  let name = if given == "" { prefix-str } else if prefix-str == "" {
    given
  } else { given + " " + prefix-str }
  let display = if name == "" { family } else if family == "" { name } else {
    name + " " + family
  }
  if suffix != "" {
    display = if display == "" { suffix } else { display + ", " + suffix }
  }
  (raw: raw, display: display)
}

// ---- names --------------------------------------------------------------
// The value of a person field: always a list of persons.
#let _persons-of(entry, field) = {
  let v = entry.at(field, default: none)
  if v == none or v == "" { () } else if type(v) == array { v } else { (v,) }
}

// Comma-and list of display names; more than `et-al` names are truncated
// with an italic "et al" (`et-al: none` lists everyone).
#let _name-list(persons, et-al) = {
  let names = persons.map(p => _person(p).display).filter(n => n != "")
  if names.len() == 0 { none } else if et-al != none and names.len() > et-al {
    names.slice(0, et-al).join(", ") + [, #emph[et al]]
  } else if names.len() == 1 { names.first() } else if names.len() == 2 {
    names.first() + " and " + names.last()
  } else { names.slice(0, -1).join(", ") + ", and " + names.last() }
}

// ---- dates --------------------------------------------------------------
// The 4-digit year group of an ISO date (YYYY[-MM[-DD]]), sign included.
#let _year-of(entry) = {
  let m = str(entry.at("date", default: "")).match(regex("-?\\d{4}"))
  if m == none { "" } else { m.text }
}

// Last two digits of the year, for the citation label.
#let _year(entry) = {
  let y = _year-of(entry)
  if y == "" { "" } else { y.slice(-2) }
}

// Full year for display; negative years render as "B.C.E.".
#let _full-year(entry) = {
  let y = _year-of(entry)
  if y == "" { "" } else if y.starts-with("-") {
    str(calc.abs(int(y))) + " B.C.E."
  } else { y }
}

// ---- labels -------------------------------------------------------------
#let _initial(s) = if s == "" { "" } else { s.slice(0, 1) }

#let _tag(entry) = {
  let creators = _persons-of(entry, "author")
  if creators.len() == 0 { creators = _persons-of(entry, "editor") }
  let surnames = creators.map(p => _person(p).raw)
  if surnames.len() == 0 {
    "Unknown"
  } else if surnames.len() == 1 {
    surnames.first().slice(0, calc.min(3, surnames.first().len()))
  } else if surnames.len() <= 4 {
    surnames.map(_initial).join()
  } else {
    surnames.slice(0, 3).map(_initial).join() + "+"
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

// ---- small helpers ------------------------------------------------------
#let _pages(s) = {
  if s == none { none } else { str(s).split("-").map(x => x.trim()).join("–") }
}

#let _vol-issue(parent) = {
  let vol = parent.at("volume", default: none)
  let issue = parent.at("issue", default: none)
  if vol == none { none } else {
    str(vol) + if issue == none { "" } else { "(" + str(issue) + ")" }
  }
}

// "3" -> "3rd", for numeric editions.
#let _ordinal(n) = {
  let n = int(n)
  let rem100 = calc.rem(n, 100)
  let rem10 = calc.rem(n, 10)
  let suffix = if rem100 >= 11 and rem100 <= 13 { "th" } else if rem10 == 1 {
    "st"
  } else if rem10 == 2 { "nd" } else if rem10 == 3 { "rd" } else { "th" }
  str(n) + suffix
}

// A publisher: plain string or {name, location} mapping.
#let _publisher-str(p) = {
  if p == none { "" } else if type(p) == dictionary {
    let name = _fs-str(p.at("name", default: ""))
    let loc = _fs-str(p.at("location", default: ""))
    if name == "" { loc } else if loc == "" { name } else { name + ", " + loc }
  } else {
    _fs-str(p)
  }
}

// The recognized serial numbers of the `serial-number` field, stringified.
#let _serial-nums(entry) = {
  let sn = entry.at("serial-number", default: none)
  if type(sn) == dictionary {
    let out = (:)
    for k in ("doi", "arxiv", "serial", "isbn", "issn", "pmid", "pmcid") {
      let v = sn.at(k, default: none)
      if v != none { out.insert(k, str(v)) }
    }
    out
  } else if type(sn) == str { (serial: str(sn)) } else { (:) }
}

#let _serial-labels = (
  isbn: "ISBN ",
  issn: "ISSN ",
  pmid: "PMID ",
  pmcid: "PMCID ",
)

// ---- parents ------------------------------------------------------------

#let _parents-of(entry) = {
  let p = entry.at("parent", default: none)
  if type(p) == array { p } else if p == none { () } else { (p,) }
}

// Default parent types of entry types that define one (spec, "Representing
// publication circumstance with parents").
#let _default-parent = (
  article: "Periodical",
  chapter: "Book",
  entry: "Reference",
  anthos: "Anthology",
  web: "Web",
  scene: "Video",
  artwork: "Exhibition",
  legislation: "Anthology",
  video: "Video",
  audio: "Audio",
  post: "Post",
)

// Venue segments for one parent of `entry`. `depth` is the nesting level
// within its parent chain (an outer container renders series-like);
// `primary` marks the first parent of the first level, the one that owns
// the page range and the full venue description.
#let _parent-segs(parent, entry, entry-type, depth, primary) = {
  let ptype = lower(_fs-str(parent.at("type", default: "")))
  if ptype == "" { ptype = lower(_default-parent.at(entry-type, default: "")) }
  let title-str = _fs-str(parent.at("title", default: ""))
  let title-c = _fs-content(parent.at("title", default: ""))
  let segs = ()
  if primary and ptype == "periodical" {
    if title-str != "" { segs.push(emph(title-c)) }
    let vi = _vol-issue(parent)
    let end = _pages(entry.at("page-range", default: none))
    if end == none {
      // Without a page range the serial is the article number, rendered
      // as volume(issue):serial (e.g. ACM "Art. no. 3").
      let art = _serial-nums(entry).at("serial", default: none)
      if art != none { end = art }
    }
    if vi != none {
      segs.push(str(vi) + if end == none { "" } else { ":" + end })
    } else if end != none { segs.push(end) }
  } else {
    // Proceedings, Conference, Book, Series, Anthology, Video, ...
    if title-str != "" {
      segs.push(if depth == 0 { [In #emph(title-c)] } else { emph(title-c) })
    } else if depth == 0 and ptype != "" {
      // Title-less parents are described by their type.
      segs.push(ptype)
    }
    if depth == 0 {
      let eds = _name-list(_persons-of(parent, "editor"), none)
      if eds != none { segs.push("edited by " + eds) }
      let pub = _publisher-str(parent.at("publisher", default: none))
      if pub != "" { segs.push(pub) }
      let org = _fs-str(parent.at("organization", default: ""))
      if org != "" { segs.push(org) }
      let loc = _fs-str(parent.at("location", default: ""))
      if loc != "" { segs.push(loc) }
    }
    if parent.at("volume", default: none) != none {
      segs.push(str(parent.volume))
    }
    if primary and entry.at("page-range", default: none) != none {
      segs.push([pages #_pages(entry.page-range)])
    }
  }
  for f in ("runtime", "time-range") {
    let v = parent.at(f, default: none)
    if v != none { segs.push(str(v)) }
  }
  segs
}

// ---- URLs ---------------------------------------------------------------
// Canonical URLs of the entry and all its parents, each with its optional
// access date.
#let _urls(entry) = {
  let out = ()
  let queue = (entry,)
  for p in _parents-of(entry) { queue.push(p) }
  for e in queue {
    let u = e.at("url", default: none)
    let value = if type(u) == dictionary { u.at("value", default: none) } else {
      u
    }
    if value != none {
      let accessed = if type(u) == dictionary {
        u.at("date", default: none)
      } else { none }
      out.push((
        str(value),
        if accessed == none { none } else { str(accessed) },
      ))
    }
  }
  out
}

// DOI / arXiv links, or the canonical URLs when neither is present.
// The serial number (ePrint number, RFC, article number) is rendered in the
// entry body instead, see `_entry-body`.
#let _suffix(entry) = {
  let sn = _serial-nums(entry)
  let parts = ()
  if sn.at("doi", default: none) != none {
    parts.push(link("https://doi.org/" + sn.doi, [doi: #sn.doi]))
  }
  if sn.at("arxiv", default: none) != none {
    let urls = _urls(entry)
    let target = if urls.len() > 0 { urls.first().at(0) } else {
      "https://arxiv.org/abs/" + sn.arxiv
    }
    parts.push(link(target, [arXiv: #sn.arxiv]))
  }
  if parts.len() == 0 {
    for (u, accessed) in _urls(entry) {
      parts.push(if accessed == none { link(u, [#u]) } else {
        [#link(u, [#u]), accessed #accessed]
      })
    }
  }
  if parts.len() == 0 { none } else { parts.join(", ") }
}

// ---- affiliated persons -------------------------------------------------
#let _role-phrases = (
  translator: "translated by",
  afterword: "afterword by",
  foreword: "foreword by",
  introduction: "introduction by",
  annotator: "annotated by",
  commentator: "commentary by",
  holder: "held by",
  compiler: "compiled by",
  founder: "founded by",
  collaborator: "in collaboration with",
  organizer: "organized by",
  "cast-member": "with",
  composer: "music by",
  producer: "produced by",
  "executive-producer": "executive produced by",
  writer: "written by",
  cinematography: "cinematography by",
  director: "directed by",
  illustrator: "illustrated by",
  narrator: "narrated by",
)

// The `affiliated` field: (role, names) mappings, rendered as role phrases.
#let _affiliated-segs(entry) = {
  let aff = entry.at("affiliated", default: none)
  let list = if type(aff) == array { aff } else if aff == none { () } else {
    (aff,)
  }
  let segs = ()
  for a in list {
    let names = a.at("names", default: ())
    let persons = if type(names) == array { names } else { (names,) }
    if persons.len() > 0 {
      let role = lower(str(a.at("role", default: "")))
      let phrase = _role-phrases.at(role, default: "")
      segs.push(if phrase == "" { _name-list(persons, none) } else {
        phrase + " " + _name-list(persons, none)
      })
    }
  }
  segs
}

// ---- entry body ---------------------------------------------------------

// Entry types whose title is a work inside a container get quotation marks.
#let _quoted-types = (
  "article",
  "chapter",
  "entry",
  "anthos",
  "scene",
  "misc",
  "post",
  "thread",
  "manuscript",
)

#let _entry-body(entry, et-al, show-notes) = {
  let typ = lower(_fs-str(entry.at("type", default: "misc")))
  if typ == "" { typ = "misc" }

  let authors = _persons-of(entry, "author")
  let editors = _persons-of(entry, "editor")
  let creator = _name-list(authors, et-al)
  if creator == none and editors.len() > 0 {
    creator = "edited by " + _name-list(editors, et-al)
  }

  // A truncated list ends with italic "et al" (the ". " segment joiner
  // supplies the dot after it).
  let quoted = typ in _quoted-types
  let title-str = _fs-str(entry.at("title", default: ""))
  let title-c = _fs-content(entry.at("title", default: ""))
  // A title that already ends in ., ! or ? is not given a second terminator.
  let dot = if title-str.match(regex("[.!?]$")) == none { [.] } else { [] }
  let title-seg = if quoted { [“#title-c#dot” ] } else { [#title-c#dot ] }

  // Container bits; the serial number and year are appended commonly below.
  let tail = ()
  for (i, p) in _parents-of(entry).enumerate() {
    // Guard against (malformed) parent cycles.
    let cur = p
    let depth = 0
    while cur != none and depth < 8 {
      tail += _parent-segs(cur, entry, typ, depth, i == 0 and depth == 0)
      cur = cur.at("parent", default: none)
      depth += 1
    }
  }

  // Entry-scoped publication facts of standalone works.
  let ed = entry.at("edition", default: none)
  if ed != none {
    let s = str(ed).trim()
    tail.push(if s.match(regex("^\\d+$")) != none {
      _ordinal(s) + " ed."
    } else { s })
  }
  if entry.at("chapter", default: none) != none {
    tail.push("chap. " + str(entry.chapter))
  }
  if entry.at("volume", default: none) != none {
    tail.push("vol. " + str(entry.volume))
  }
  for f in ("runtime", "time-range") {
    if entry.at(f, default: none) != none { tail.push(str(entry.at(f))) }
  }
  let has-parent-title = {
    let ps = _parents-of(entry)
    ps.len() > 0 and _fs-str(ps.first().at("title", default: "")) != ""
  }
  if not has-parent-title {
    // Standalone works name their publisher, institution, and place.
    let pub = _publisher-str(entry.at("publisher", default: none))
    if pub != "" { tail.push(pub) }
    let genre = _fs-str(entry.at("genre", default: ""))
    if genre != "" { tail.push(genre) }
    let org = _fs-str(entry.at("organization", default: ""))
    if org != "" { tail.push(org) }
    let loc = _fs-str(entry.at("location", default: ""))
    if loc != "" { tail.push(loc) }
  }
  for f in ("archive", "archive-location", "call-number") {
    let v = _fs-str(entry.at(f, default: ""))
    if v != "" { tail.push(v) }
  }
  tail += _affiliated-segs(entry)

  let sn = _serial-nums(entry)
  // In journals without a page range the serial was consumed as the article
  // number above.
  let parents = _parents-of(entry)
  let p0type = if parents.len() > 0 {
    lower(_fs-str(parents.first().at("type", default: "")))
  } else { "" }
  if p0type == "" { p0type = lower(_default-parent.at(typ, default: "")) }
  let consumed-as-artno = (
    p0type == "periodical" and entry.at("page-range", default: none) == none
  )
  for k in ("serial", "isbn", "issn", "pmid", "pmcid") {
    let v = sn.at(k, default: none)
    if v != none and not (consumed-as-artno and k == "serial") {
      tail.push(_serial-labels.at(k, default: "") + v)
    }
  }

  let dy = _full-year(entry)
  if dy != "" { tail.push(dy) }

  if show-notes {
    let note = _fs-str(entry.at("note", default: ""))
    if note != "" { tail.push(note) }
  }

  // An empty tail joins to `none` (not to ""), which would sneak a stray
  // comma in front of the suffix below; `_join` guards against that.
  let rest = _join(tail, ", ")
  let suf = _suffix(entry)

  // Segments after the title are comma-joined and closed by one period
  // ("... Journal, 60:113-119, 2014, doi: ..."). A segment that already
  // ends in a period ("Jr.", "B.C.E.") gets no second one.
  let segs = ()
  if creator != none {
    let c-str = if type(creator) == str { creator } else { "" }
    let c-dot = if c-str.match(regex("[.]$")) == none { [.] } else { [] }
    segs.push([#creator#c-dot ])
  }
  segs.push(title-seg)
  if rest != "" or suf != none {
    let bits = ()
    if rest != "" { bits.push(rest) }
    if suf != none { bits.push(suf) }
    // Without a link suffix the last tail segment can already end in a
    // period ("..., 500 B.C.E."); with one, the period always follows.
    let last-str = if suf != none { "" } else if tail.len() > 0 {
      tail.last()
    } else { "" }
    let end-dot = if (
      type(last-str) == str and last-str.match(regex("[.]$")) != none
    ) { [] } else { [.] }
    segs.push(bits.join(", ") + end-dot)
  }
  segs.join("")
}

// Bind `cite`, `citesec` and `bibliography` to the given bibliography
// entries.
//
// The entries are loaded by the caller, so the yaml path is resolved
// relative to the caller's own file, with no hidden coupling to lib/:
//
//   #import "lib/bibliography.typ": init-bib
//   #let (cite, citesec, bibliography) = init-bib(yaml("BIBLIOGRAPHY.yml"))
//
//   #cite(<harvey2014>)   // [Har14], links to the bibliography entry
//   #citesec(<harvey2014>, "3")  // [Har14, § 3]
//   #bibliography(title: "Bibliography")
//
// With more than `et-al` authors, the bibliography lists the first `et-al`
// names followed by "et al."; pass a larger value or `none` to list everyone.
//
// Like Typst's built-in bibliography, only cited entries are shown unless
// `full: true`. Labels are computed once over all entries, so `cite` and
// `bibliography` always agree, also for collision suffixes.
#let init-bib(entries, et-al: 6, show-notes: false) = {
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

  // Like `cite`, but appends a section of the referenced work:
  // `citesec(<emvp2025>, "7.4")` renders [BCH+25, § 7.4].
  let citesec(key, section) = {
    if str(key) not in labels {
      panic("no bibliography entry for key", str(key))
    }
    let name = str(key)
    let rendered = "[" + link(key, labels.at(name)) + ", § " + section + "]"
    [#rendered#metadata((name,)) <alpha-bib-cited>]
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
          ([#labels.at(k)#label(k)], _entry-body(p.at(1), et-al, show-notes))
        })
        .flatten(),
    )
  }

  (cite: cite, citesec: citesec, bibliography: bibliography)
}
