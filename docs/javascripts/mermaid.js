document$.subscribe(function () {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: true,
  });

  for (const element of document.querySelectorAll(".mermaid")) {
    const source = element.textContent;
    if (source) {
      element.removeAttribute("data-processed");
      element.innerHTML = source;
    }
  }

  mermaid.run({
    querySelector: ".mermaid",
  });
});
