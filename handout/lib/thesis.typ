#let thesis(
  dedication: none,
  font: none,
  toc: true,
  lang: "en",
  doc,
) = {
  set page(
    paper: "a4",
    margin: (x: 2.5cm, y: 2.5cm),
    numbering: "1",
    footer: align(center, context text(10pt, counter(page).display("1"))),
  )
  set text(
    font: if (font != none) { font } else {
      ("Arial", "Liberation Sans", "Verdana", "Tahoma", "DejaVu
    Sans")
    },
    size: 12pt,
    lang: lang,
  )
  // 1.4em leading keeps a full page at ~28 lines (guide: 26-30)
  set par(justify: true, leading: 1.3em, spacing: 2.4em)
  set heading(numbering: "1.1")
  show heading: set block(above: 2.6em, below: 1.3em)

  page(numbering: none, footer: none)[]
  page(numbering: none, footer: none)[]
  page(numbering: none, footer: none)[
    #if dedication != none {
      align(center + horizon, emph(text(13pt, dedication)))
    }
  ]
  page(numbering: none, footer: none)[]

  counter(page).update(1)

  if toc {
    outline(depth: 3)
    pagebreak()
  }

  doc
}
