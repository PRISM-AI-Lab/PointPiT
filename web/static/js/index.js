"use strict";

const scrollButton = document.querySelector(".scroll-to-top");
const copyButton = document.querySelector(".copy-bibtex-btn");

window.addEventListener(
  "scroll",
  () => {
    scrollButton.classList.toggle("visible", window.scrollY > 300);
  },
  { passive: true },
);

scrollButton.addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
});

copyButton.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex-code").textContent;
  const label = copyButton.querySelector(".copy-text");

  try {
    await navigator.clipboard.writeText(citation);
  } catch (error) {
    const textArea = document.createElement("textarea");
    textArea.value = citation;
    textArea.setAttribute("readonly", "");
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    textArea.remove();
  }

  copyButton.classList.add("copied");
  label.textContent = "Copied!";
  window.setTimeout(() => {
    copyButton.classList.remove("copied");
    label.textContent = "Copy";
  }, 2000);
});
