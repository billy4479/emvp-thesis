// Self-asserting test suite for `bibliography.typ`, targeting the full
// Hayagriva YAML file format
// (https://github.com/typst/hayagriva/blob/main/docs/file-format.md).
//
// Run with: typst compile handout/lib/bibliography-tests.typ
// A failed assertion aborts the compilation with a pointed error.
//
// Layout: unit tests first (pure helpers), then two bibliographies rendered
// from inline fixtures and checked end-to-end. The `#show text` rule turns
// every text element into queryable metadata, so the rendered bibliography
// text can be compared against expectations (whitespace-insensitively).
//
// The suite is split in two because "does not contain" assertions are only
// sound within one rendering: the main bibliography uses `et-al: none` and
// hides notes; the auxiliary one uses `et-al: 3` and `show-notes: true`.
// The aux rendering is the last element of the document, so its text can be
// isolated at its heading.

#import "bibliography.typ": init-bib
#import "bibliography.typ": (
  _affiliated-segs, _default-parent, _fs-content, _fs-str, _full-year,
  _name-list, _ordinal, _pages, _person, _persons-of, _publisher-str,
  _serial-nums, _vol-issue, _year, _year-of,
)

// ---- unit tests: formattable strings ------------------------------------

#assert.eq(_fs-str("plain"), "plain")
#assert.eq(_fs-str(none), "")
#assert.eq(_fs-str(2026), "2026")
#assert.eq(_fs-str((value: "verbatim value", verbatim: true)), "verbatim value")
#assert.eq(
  _fs-str((value: "International Proceedings", short: "Int. Proc.")),
  "International Proceedings",
)
#assert.eq(_fs-str((value: (value: "nested"), verbatim: true)), "nested")

// Balanced dollars become math; unbalanced dollars stay plain text. The
// math branch is exercised end-to-end below (the "$" must not survive).
#assert.eq(_fs-str("On $x^2$"), "On $x^2$")
#assert.eq(_fs-content("no math"), "no math")
#assert(
  _fs-content("On $x^2$") != "On $x^2$",
  message: "math must become content",
)

// ---- unit tests: persons ------------------------------------------------

#assert.eq(_person("Doe, Janet").display, "Janet Doe")
#assert.eq(_person("Doe, Janet").raw, "Doe")
// "Family, Suffix, Given"
#assert.eq(
  _person("Luther King, Jr., Martin").display,
  "Martin Luther King, Jr.",
)
#assert.eq(_person("Luther King, Jr., Martin").raw, "Luther King")
// Organizations pass through.
#assert.eq(_person("UNICEF").display, "UNICEF")
#assert.eq(_person("UNICEF").raw, "UNICEF")
// BibTeX rule: lowercase leading words are the name prefix.
#assert.eq(_person("von der Leyen, Ursula").display, "Ursula von der Leyen")
#assert.eq(_person("von der Leyen, Ursula").raw, "von der Leyen")
// Dictionary persons.
#assert.eq(
  _person((
    given-name: "Gloria Jean",
    name: "Watkins",
    alias: "bell hooks",
  )).display,
  "Gloria Jean Watkins",
)
#assert.eq(
  _person((
    given-name: "Gloria Jean",
    name: "Watkins",
    alias: "bell hooks",
  )).raw,
  "Watkins",
)
#assert.eq(
  _person((name: "Leyen", prefix: "von der", given-name: "Ursula")).display,
  "Ursula von der Leyen",
)
#assert.eq(
  _person((
    name: "Leyen",
    prefix: "von der",
    given-name: "Ursula",
    suffix: "Jr.",
  )).display,
  "Ursula von der Leyen, Jr.",
)

// ---- unit tests: name lists --------------------------------------------

#let two = ("Doe, Jane", "Ray, Kate")
#let four = ("Doe, Jane", "Ray, Kate", "Low, Ada", "Nut, Bea")
#assert.eq(_name-list((), none), none)
#assert.eq(_name-list((two.first(),), none), "Jane Doe")
#assert.eq(_name-list(two, none), "Jane Doe and Kate Ray")
#assert.eq(_name-list(four, none), "Jane Doe, Kate Ray, Ada Low, and Bea Nut")

// ---- unit tests: dates --------------------------------------------------

#assert.eq(_year-of((date: "2018-06")), "2018")
#assert.eq(_year-of((date: "2011-11-08")), "2011")
#assert.eq(_year-of((date: "2020")), "2020")
#assert.eq(_year-of((date: "-0050")), "-0050")
#assert.eq(_year-of((date: "n.d.")), "")
#assert.eq(_year((date: "2018-06")), "18")
#assert.eq(_year((:)), "")
#assert.eq(_full-year((date: "-0050")), "50 B.C.E.")
#assert.eq(_full-year((date: "2020")), "2020")
#assert.eq(_full-year((:)), "")

// ---- unit tests: small helpers ------------------------------------------

#assert.eq(_pages("311-334"), "311–334")
#assert.eq(_pages(113), "113")
#assert.eq(_pages(none), none)
#assert.eq(_vol-issue((volume: 29, issue: 1)), "29(1)")
#assert.eq(_vol-issue((volume: 60)), "60")
#assert.eq(_vol-issue((:)), none)
#assert.eq(_ordinal(1), "1st")
#assert.eq(_ordinal(2), "2nd")
#assert.eq(_ordinal(3), "3rd")
#assert.eq(_ordinal(4), "4th")
#assert.eq(_ordinal(11), "11th")
#assert.eq(_ordinal(13), "13th")
#assert.eq(_ordinal(21), "21st")
#assert.eq(_ordinal(111), "111th")
#assert.eq(_ordinal(123), "123rd")
#assert.eq(_publisher-str("Penguin Books"), "Penguin Books")
#assert.eq(
  _publisher-str((name: "Penguin Books", location: "London, UK")),
  "Penguin Books, London, UK",
)
#assert.eq(_publisher-str((location: "London, UK")), "London, UK")
#assert.eq(_publisher-str(none), "")
#assert.eq(_serial-nums((serial-number: "RFC 8439")), (serial: "RFC 8439"))
#assert.eq(_serial-nums((serial-number: (doi: "10.1/x", isbn: "isbn-value"))), (
  doi: "10.1/x",
  isbn: "isbn-value",
))
#assert.eq(_serial-nums((serial-number: none)), (:))
#assert.eq(_serial-nums((:)), (:))

// ---- unit tests: default parent types -----------------------------------

#assert.eq(_default-parent.at("article"), "Periodical")
#assert.eq(_default-parent.at("chapter"), "Book")
#assert.eq(_default-parent.at("misc", default: none), none)

// ---- unit tests: affiliated role phrases --------------------------------

#assert.eq(
  repr(_affiliated-segs((
    affiliated: (role: "Director", names: "Cameron, James"),
  ))),
  repr(("directed by James Cameron",)),
)
#assert.eq(
  repr(_affiliated-segs((
    affiliated: (
      (role: "translator", names: ("A, B", "C, D")),
      (role: "whatever", names: "X, Y"),
    ),
  ))),
  repr(("translated by B A and D C", "Y X")),
)

// ---- fixtures ------------------------------------------------------------

#let fixture = (
  // Person string with a suffix.
  suffix-person: (
    type: "Misc",
    title: "Suffix Test",
    author: "Luther King, Jr., Martin",
    date: "1963",
    note: "hidden note",
  ),
  // Person dictionary with an alias.
  dict-person: (
    type: "Misc",
    title: "Alias Test",
    author: (given-name: "Gloria Jean", name: "Watkins", alias: "bell hooks"),
    date: "1992",
  ),
  // Lowercase name prefixes, numeric edition, publisher dict, ISBN.
  prefix-name: (
    type: "Book",
    title: "Modern Computer Algebra",
    author: ("von zur Gathen, Joachim", "Gerhard, Jurgen"),
    edition: "3",
    publisher: (name: "Cambridge University Press", location: "Cambridge"),
    serial-number: (isbn: "978-1-107-03903-2"),
    date: "2013",
  ),
  // Editors of the parent, chapter number, page range.
  edited-chapter: (
    type: "Chapter",
    title: "An Edited Chapter",
    author: "Author, Anne",
    editor: ("Shannon, C. E.", "McCarthy, J."),
    chapter: 4,
    page-range: "55-70",
    date: "1956",
    parent: (
      type: "Anthology",
      title: "Automata Studies",
      editor: "Shannon, C. E.",
    ),
  ),
  // No author: the editors drive both label and body.
  edited-only: (
    type: "Book",
    title: "Collected Papers",
    editor: ("Editor, Edna", "Editor, Bob"),
    date: "2001",
  ),
  // Multiple parents (spec example): conference + video with runtime/URL.
  multi-parent: (
    type: "Article",
    title: "Boost Performance and Security with Modern Networking",
    author: ("Mehta, Jiten", "Kinnear, Eric"),
    date: "2020-06-26",
    parent: (
      (
        type: "Conference",
        title: "World Wide Developer Conference 2020",
        organization: "Apple Inc.",
        location: "Mountain View, CA",
      ),
      (
        type: "Video",
        runtime: "00:13:42",
        url: "https://developer.apple.com/videos/play/wwdc2020/10111/",
      ),
    ),
  ),
  // Nested parents: journal within a series.
  nested-parent: (
    type: "Article",
    title: "Nested parent test",
    author: "Doe, Jane",
    date: "2001",
    parent: (
      type: "Periodical",
      title: "Journal of Tests",
      volume: 5,
      issue: 2,
      parent: (type: "Series", title: "Series of Journals", volume: 12),
    ),
  ),
  // URL with access date.
  url-dict: (
    type: "Web",
    title: "URL access-date test",
    author: "Someone, Sam",
    url: (value: "https://example.com/", date: "2020-12-29"),
    date: "2020",
  ),
  // Math in the title, negative year.
  math-title: (
    type: "Article",
    title: "On $x^2 + y^2 = z^2$",
    author: "Pythagoras, of Samos",
    date: "-0500",
  ),
  // Parent without type: falls back to the default parent type (Periodical).
  default-parent: (
    type: "Article",
    title: "Default parent test",
    author: "Roe, Richard",
    page-range: "10-20",
    parent: (title: "Journal X", volume: 5, issue: 2),
    date: "2005",
  ),
  // Less common serial number types plus a DOI.
  serial-types: (
    type: "Reference",
    title: "Serial types test",
    author: "Int, Test",
    serial-number: (
      doi: "10.1/x",
      issn: "2049-3630",
      pmid: "12345678",
      pmcid: "PMC123",
    ),
    date: "2022",
  ),
  // No date, no serial, URL only: the old renderer left a stray comma here.
  no-comma: (
    type: "Web",
    title: "The No Comma Project",
    author: "Contributors, Some",
    url: "https://nocomma.example/",
  ),
  // Formattable string with a `value` subfield.
  verbatim-title: (
    type: "Misc",
    title: (value: "The {imagiNary} Publishing Guide", verbatim: true),
    author: "Wright, cosmo",
    date: "2020",
  ),
  // Affiliated persons with roles; case-insensitive role matching.
  affiliated-video: (
    type: "Video",
    title: "A Film",
    affiliated: (
      (role: "Director", names: "Cameron, James"),
      (role: "composer", names: ("Composer, Carla", "Composer, Bob")),
    ),
    date: "2009",
  ),
  // Archive cluster.
  archive-entry: (
    type: "Misc",
    title: "Informational plaque",
    author: "Museum, The",
    archive: "Landesmuseum Koblenz",
    archive-location: "Koblenz, Germany",
    call-number: "F16 D14",
    date: "2020",
  ),
  // Standalone media with a runtime.
  podcast: (
    type: "Audio",
    title: "Episode",
    author: "Host, Hanna",
    runtime: "01:42:21",
    date: "2024",
  ),
  // More than four authors for the "+" tag; long list for et al.
  long-list: (
    type: "Misc",
    title: "Long List",
    author: (
      "Aaaa, A1",
      "Bbbb, B2",
      "Cccc, C3",
      "Dddd, D4",
      "Eeee, E5",
      "Ffff, F6",
      "Gggg, G7",
      "Hhhh, H8",
    ),
    date: "2025",
    note: "internal note",
  ),
  // Label collision and disambiguation suffixes.
  coll-a: (type: "Misc", title: "Coll A", author: "Smith, John", date: "2000"),
  coll-b: (type: "Misc", title: "Coll B", author: "Smith, Jane", date: "2000"),
)

// ---- main rendering: et-al none, notes hidden ----------------------------

#let (cite, citesec, bibliography) = init-bib(fixture, et-al: none)

#show text: it => metadata(it.text)

#cite(<coll-a>, <coll-b>)
#citesec(<coll-a>, "7.4")

#bibliography(title: "Test Bibliography", full: true)

// ---- auxiliary rendering: et al truncation, notes shown ------------------

#let (cite: cite-aux, bibliography: bibliography-aux) = init-bib(
  (long-list: fixture.long-list),
  et-al: 3,
  show-notes: true,
)

#bibliography-aux(title: "Aux Bib", full: true)

// ---- end-to-end assertions ----------------------------------------------

#context {
  let texts = query(metadata).map(m => m.value).filter(v => type(v) == str)
  let all = texts.join("")
  let t = all.replace(" ", "")
  let idx = texts.position(x => x == "Aux Bib")
  assert(idx != none, message: "aux heading not found")
  let aux-t = texts.slice(idx + 1).join().replace(" ", "")

  // Citations render alpha labels; collision suffixes disambiguate.
  assert(t.contains("[Smi00,Smi00a]"), message: "cite labels: " + t)
  assert(t.contains("[Smi00,§7.4]"), message: "citesec: " + t)

  // Person suffix string; no second period after the "Jr.".
  assert(
    t.contains("MartinLutherKing,Jr.“SuffixTest.”"),
    message: "person suffix: " + t,
  )
  assert(t.contains("Jr..") == false, message: "no double period after Jr.")
  // Person dictionary with alias.
  assert(t.contains("GloriaJeanWatkins"), message: "person dict: " + t)
  // BibTeX prefix rule keeps the vG-style tag.
  assert(t.contains("vG13"), message: "prefix tag: " + t)
  // Numeric edition, publisher dict, ISBN.
  assert(
    t.contains("ModernComputerAlgebra.CambridgeUniversityPress") == false,
    message: "edition must precede publisher: " + t,
  )
  assert(
    t.contains(
      "3rded.,CambridgeUniversityPress,Cambridge,ISBN978-1-107-03903-2,2013.",
    ),
    message: "edition/publisher/isbn: " + t,
  )
  // Parent editors, chapter number, page range with en-dash.
  assert(
    t.contains("InAutomataStudies,editedbyC.E.Shannon,pages55–70,chap.4,1956."),
    message: "edited chapter: " + t,
  )
  // Authorless entries fall back to their editors.
  assert(t.contains("EE01"), message: "editor label: " + t)
  assert(
    t.contains("editedbyEdnaEditorandBobEditor."),
    message: "editor body: " + t,
  )
  // Multiple parents: conference venue, organization, location, video, runtime, URL.
  assert(
    t.contains(
      "InWorldWideDeveloperConference2020,AppleInc.,MountainView,CA,video,00:13:42,2020",
    ),
    message: "multi parent: " + t,
  )
  assert(
    t.contains("https://developer.apple.com/videos/play/wwdc2020/10111/"),
    message: "parent url: " + t,
  )
  // Nested parents: journal then series.
  assert(
    t.contains("JournalofTests,5(2),SeriesofJournals,12,2001."),
    message: "nested parent: " + t,
  )
  // URL access date.
  assert(t.contains("accessed2020-12-29"), message: "access date: " + t)
  // Math title: dollars evaluated, none survive as text.
  assert(all.contains("$") == false, message: "math must be evaluated")
  // Negative year; no second period after "B.C.E.".
  assert(t.contains("500B.C.E."), message: "bce year: " + t)
  assert(
    t.contains("B.C.E..") == false,
    message: "no double period after B.C.E.",
  )
  // Default parent type for a typeless parent.
  assert(
    t.contains("JournalX,5(2):10–20,2005."),
    message: "default parent: " + t,
  )
  // Serial number types.
  assert(
    t.contains("ISSN2049-3630,PMID12345678,PMCIDPMC123,2022,doi:10.1/x."),
    message: "serial types: " + t,
  )
  // No stray comma before the URL.
  assert(
    t.contains("TheNoCommaProject.https://nocomma.example/."),
    message: "no comma: " + t,
  )
  assert(
    t.contains(",https://nocomma") == false,
    message: "stray comma regression",
  )
  // Formattable string with value subfield, verbatim casing.
  assert(
    t.contains("The{imagiNary}PublishingGuide"),
    message: "verbatim title: " + t,
  )
  // Affiliated role phrases, case-insensitive roles.
  assert(t.contains("directedbyJamesCameron"), message: "affiliated: " + t)
  assert(
    t.contains("musicbyCarlaComposerandBobComposer"),
    message: "affiliated list: " + t,
  )
  // Archive cluster.
  assert(
    t.contains("LandesmuseumKoblenz,Koblenz,Germany,F16D14,2020"),
    message: "archive: " + t,
  )
  // Standalone runtime.
  assert(t.contains("01:42:21,2024"), message: "runtime: " + t)
  // Full author list with et-al: none.
  assert(t.contains("G7Gggg,andH8Hhhh."), message: "et-al none: " + t)
  // Five-plus authors get the "+" tag.
  assert(t.contains("ABC+25"), message: "plus tag: " + t)
  // Notes are hidden by default.
  assert(t.contains("hiddennote") == false, message: "notes hidden by default")
  // Quoted titles for work-in-container types, unquoted for standalone.
  assert(t.contains("“SuffixTest.”"), message: "quoted misc: " + t)
  assert(t.contains("“ModernComputer") == false, message: "book title unquoted")

  // Auxiliary rendering: truncation and notes.
  assert(
    aux-t.contains("A1Aaaa,B2Bbbb,C3Cccc,etal."),
    message: "et-al truncation: " + aux-t,
  )
  assert(
    aux-t.contains("Hhhh") == false,
    message: "truncated list must drop later authors",
  )
  assert(aux-t.contains("internalnote"), message: "show-notes: " + aux-t)
}
