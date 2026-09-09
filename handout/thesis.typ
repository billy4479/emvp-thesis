#import "lib/thesis.typ": thesis
#import "lib/alpha-bib.typ": init-bib

#show: thesis.with(
  dedication: "Dedication or acknowledgements",
  // font: "New Computer Modern",
  font: "",
)

#let (cite, bibliography) = init-bib(yaml("BIBLIOGRAPHY.yml"))

= Introduction

#lorem(100)

#cite(<emvp2025>)

#bibliography(title: "Bibliography")

