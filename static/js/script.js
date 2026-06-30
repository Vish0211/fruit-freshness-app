const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const trayEmpty = document.getElementById('trayEmpty');
const trayPreview = document.getElementById('trayPreview');
const previewImg = document.getElementById('previewImg');

const resultIdle = document.getElementById('resultIdle');
const resultLoading = document.getElementById('resultLoading');
const resultData = document.getElementById('resultData');
const resultError = document.getElementById('resultError');
const errorText = document.getElementById('errorText');

const stamp = document.getElementById('stamp');
const fruitName = document.getElementById('fruitName');
const gaugeArc = document.getElementById('gaugeArc');
const gaugeNeedle = document.getElementById('gaugeNeedle');
const confidenceValue = document.getElementById('confidenceValue');
const breakdown = document.getElementById('breakdown');
const resetBtn = document.getElementById('resetBtn');

const ARC_LENGTH = 267; // matches the path length in the SVG

function showPanel(panel) {
  [resultIdle, resultLoading, resultData, resultError].forEach(p => {
    p.hidden = (p !== panel);
  });
}

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

resetBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.value = '';
  trayEmpty.hidden = false;
  trayPreview.hidden = true;
  showPanel(resultIdle);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showPanel(resultError);
    errorText.textContent = 'That file is not an image. Please upload a PNG, JPG, or WEBP.';
    return;
  }

  // Show the preview immediately
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    trayEmpty.hidden = true;
    trayPreview.hidden = false;
  };
  reader.readAsDataURL(file);

  // Send to the server
  showPanel(resultLoading);

  const formData = new FormData();
  formData.append('image', file);

  fetch('/predict', { method: 'POST', body: formData })
    .then(res => res.json().then(data => ({ ok: res.ok, data })))
    .then(({ ok, data }) => {
      if (!ok) {
        throw new Error(data.error || 'Prediction failed.');
      }
      renderResult(data);
    })
    .catch(err => {
      showPanel(resultError);
      errorText.textContent = err.message;
    });
}

function renderResult(data) {
  const isFresh = data.freshness.toLowerCase() === 'fresh';

  stamp.textContent = data.freshness.toUpperCase();
  stamp.className = 'stamp ' + (isFresh ? 'fresh' : 'rotten');
  fruitName.textContent = data.fruit;

  // Color the gauge to match the verdict
  const color = isFresh ? 'var(--fresh)' : 'var(--rotten)';
  gaugeArc.parentElement.style.color = color;

  // Animate the arc fill based on confidence (0-100 -> 0-ARC_LENGTH offset)
  const offset = ARC_LENGTH - (ARC_LENGTH * data.confidence / 100);
  requestAnimationFrame(() => {
    gaugeArc.style.strokeDashoffset = offset;
  });

  // Needle sweeps from -90deg (left) to +90deg (right) across the dial
  const angle = -90 + (180 * data.confidence / 100);
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;

  // Animate the number count-up
  animateNumber(confidenceValue, data.confidence);

  // Build the per-class breakdown bars
  breakdown.innerHTML = '';
  data.all_scores.slice(0, 4).forEach(item => {
    const row = document.createElement('div');
    row.className = 'breakdown-row';
    row.innerHTML = `
      <span class="breakdown-label">${item.label}</span>
      <span class="breakdown-bar-track"><span class="breakdown-bar-fill" style="width:${item.confidence}%"></span></span>
      <span class="breakdown-pct">${item.confidence}%</span>
    `;
    breakdown.appendChild(row);
  });

  showPanel(resultData);
}

function animateNumber(el, target) {
  let current = 0;
  const step = target / 20;
  const interval = setInterval(() => {
    current += step;
    if (current >= target) {
      current = target;
      clearInterval(interval);
    }
    el.textContent = current.toFixed(1);
  }, 25);
}
