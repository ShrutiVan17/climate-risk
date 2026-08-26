const header = document.querySelector('.site-header');
const sliders = [...document.querySelectorAll('input[type="range"]')];
const intensityNode = document.querySelector('#intensity');
const intensityLabel = document.querySelector('#intensity-label');
const scenarioNote = document.querySelector('#scenario-note');
const gauge = document.querySelector('#gauge');

const names = {
  events: 'climate events',
  hazard: 'hazard severity',
  duration: 'event duration'
};

function sliderFill(slider) {
  const value = Number(slider.value);
  const min = Number(slider.min);
  const max = Number(slider.max);
  const fill = ((value - min) / (max - min)) * 100;
  slider.style.setProperty('--fill', `${fill}%`);
  document.querySelector(`#${slider.id}-value`).textContent = `${value.toFixed(2)}×`;
}

function updateScenario() {
  const values = Object.fromEntries(sliders.map(slider => [slider.id, Number(slider.value)]));
  const weighted = ((values.events - 1) * .36) + ((values.hazard - 1) * .42) + ((values.duration - 1) * .22);
  const intensity = Math.max(0, Math.min(100, Math.round(20 + weighted * 92)));
  const strongest = Object.entries(values).sort((a, b) => b[1] - a[1])[0][0];
  const label = intensity < 20 ? 'Calm' : intensity < 45 ? 'Elevated' : intensity < 70 ? 'Severe' : 'Extreme';

  intensityNode.textContent = intensity;
  intensityLabel.textContent = label;
  gauge.style.setProperty('--gauge', `${intensity}%`);
  scenarioNote.textContent = `${label} relative pressure with ${names[strongest]} contributing the strongest lift.`;
  sliders.forEach(sliderFill);
}

window.addEventListener('scroll', () => {
  header.classList.toggle('scrolled', window.scrollY > 24);
}, { passive: true });

sliders.forEach(slider => slider.addEventListener('input', updateScenario));
updateScenario();
