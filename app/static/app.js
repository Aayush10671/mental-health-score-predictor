const form = document.querySelector('#prediction-form');
const resultEmpty = document.querySelector('#result-empty');
const resultContent = document.querySelector('#result-content');
const scoreElement = document.querySelector('#score');
const scoreFill = document.querySelector('#score-fill');
const scoreMessage = document.querySelector('#score-message');
const resetButton = document.querySelector('#reset-button');

const fallbackOptions = {
  stress_level: ['Low', 'Medium', 'High', 'Very High'],
};

function fillSelect(select, values) {
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    return option;
  }));
}

async function loadOptions() {
  try {
    const response = await fetch('/api/options');
    if (!response.ok) throw new Error('Could not load options');
    const options = await response.json();
    document.querySelectorAll('select[data-option]').forEach((select) => {
      fillSelect(select, options[select.dataset.option] || fallbackOptions[select.dataset.option] || []);
    });
  } catch (error) {
    document.querySelectorAll('select[data-option]').forEach((select) => {
      fillSelect(select, fallbackOptions[select.dataset.option] || []);
    });
  }
}

function formPayload() {
  const values = Object.fromEntries(new FormData(form).entries());
  ['age', 'daily_unlocks'].forEach((key) => { values[key] = Number.parseInt(values[key], 10); });
  ['avg_daily_usage_hours', 'study_hours', 'physical_activity_hours', 'sleep_hours_per_night']
    .forEach((key) => { values[key] = Number.parseFloat(values[key]); });
  return values;
}

function showResult(score) {
  scoreElement.textContent = score.toFixed(2);
  scoreFill.style.width = `${Math.max(0, Math.min(10, score)) * 10}%`;
  scoreMessage.textContent = score >= 7
    ? 'Your inputs suggest a strong overall wellbeing pattern.'
    : score >= 5
      ? 'Your inputs suggest room for a few healthier daily patterns.'
      : 'Your inputs suggest checking in with your daily balance.';
  resultEmpty.hidden = true;
  resultContent.hidden = false;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  button.firstElementChild.textContent = 'Calculating...';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formPayload()),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Prediction failed');
    showResult(body.mental_health_score);
  } catch (error) {
    window.alert(error.message);
  } finally {
    button.disabled = false;
    button.firstElementChild.textContent = 'Calculate my score';
  }
});

resetButton.addEventListener('click', () => {
  resultContent.hidden = true;
  resultEmpty.hidden = false;
  scoreFill.style.width = '0';
  form.querySelector('input')?.focus();
});

loadOptions();
