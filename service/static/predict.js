// Browser-side checks for the "Will You Get In?" form.
//
// These mirror service/validation.py so mistakes are caught before a round
// trip, but they are a convenience only: the server repeats every check.
// If this script fails to run, the form still submits and the server's
// error messages appear instead.

"use strict";

const TEXT_FIELDS = ["program", "university", "comments"];

function setError(input, message) {
  const errorEl = document.getElementById(input.id + "-error");
  if (errorEl) errorEl.textContent = message;
  if (message) {
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", input.id + "-error");
  } else {
    input.removeAttribute("aria-invalid");
  }
}

// Returns an error message, or "" when the value is blank or in range.
function checkNumber(input) {
  const raw = input.value.trim();
  if (raw === "") return "";
  const label = input.dataset.label;
  const min = Number(input.dataset.min);
  const max = Number(input.dataset.max);
  // Number() needs the whole string to be numeric ("3.5abc" is NaN), and
  // Number.isFinite rejects "Infinity".
  const value = Number(raw);
  if (!Number.isFinite(value)) return label + " must be a number, or left blank.";
  if (value < min || value > max) return label + " must be between " + min + " and " + max + ".";
  return "";
}

function init() {
  const form = document.querySelector(".predict-form");
  if (!form) return;
  const numberInputs = Array.from(form.querySelectorAll("input[data-min]"));
  const formError = document.getElementById("form-error");

  form.addEventListener("submit", (event) => {
    let firstBad = null;
    numberInputs.forEach((input) => {
      const message = checkNumber(input);
      setError(input, message);
      if (message && !firstBad) firstBad = input;
    });

    const anyFilled = TEXT_FIELDS.concat(numberInputs.map((i) => i.id))
      .some((id) => document.getElementById(id).value.trim() !== "");
    formError.textContent = anyFilled ? "" : "Please fill in at least one field before submitting.";
    if (!anyFilled && !firstBad) firstBad = document.getElementById("program");

    if (firstBad) {
      event.preventDefault();
      firstBad.focus();
      return;
    }
    // Prevent double submits while the model runs.
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    button.textContent = "Predicting…";
  });

  // Re-enable the button if the visitor comes back via the Back button
  // (browsers restore the page exactly as it was left, i.e. disabled).
  window.addEventListener("pageshow", () => {
    const button = form.querySelector("button[type=submit]");
    button.disabled = false;
    button.textContent = "Predict";
  });

  // Clear a field's error as soon as it's fixed.
  numberInputs.forEach((input) => {
    input.addEventListener("input", () => {
      if (!checkNumber(input)) setError(input, "");
    });
  });
}

document.addEventListener("DOMContentLoaded", init);
