const form = document.getElementById("predForm");
const resultBox = document.getElementById("result");
const btn = form.querySelector(".btn-predict");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  btn.disabled = true;
  btn.textContent = "Predicting...";
  resultBox.classList.add("hidden");

  const fields = ["age","sex","cp","trestbps","chol","fbs","restecg",
                  "thalach","exang","oldpeak","slope","ca","thal"];
  const payload = {};
  fields.forEach(f => payload[f] = document.getElementById(f).value);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    const isHighRisk = data.prediction === 1;
    resultBox.className = `result ${isHighRisk ? "high-risk" : "low-risk"}`;
    resultBox.innerHTML = `
      <h2>${isHighRisk ? "⚠️ High Risk: Heart Disease Detected" : "✅ Low Risk: No Heart Disease Detected"}</h2>
      <div class="prob-cards">
        <div class="prob-card no-disease">
          <div class="label">No Disease Probability</div>
          <div class="value">${data.prob_no_disease}%</div>
        </div>
        <div class="prob-card disease">
          <div class="label">Heart Disease Probability</div>
          <div class="value">${data.prob_disease}%</div>
        </div>
      </div>
    `;
    resultBox.classList.remove("hidden");
    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (err) {
    showError("Network error. Please try again.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict";
  }
});

function showError(msg) {
  resultBox.className = "result high-risk";
  resultBox.innerHTML = `<h2>❌ Error: ${msg}</h2>`;
  resultBox.classList.remove("hidden");
}
