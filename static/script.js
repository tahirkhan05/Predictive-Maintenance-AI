/**
 * Predictive Maintenance AI - Client-side Interaction Controller
 * Handles interactive telemetry form submission, scenario presets, stress gauges, and diagnostic modal.
 */

document.addEventListener('DOMContentLoaded', () => {
    const predictionForm = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const formError = document.getElementById('formError');

    // UI elements for prediction outputs
    const resultBanner = document.getElementById('resultBanner');
    const statusTitle = document.getElementById('statusTitle');
    const statusBadgeIcon = document.getElementById('statusBadgeIcon');
    const probabilityValue = document.getElementById('probabilityValue');
    const probabilityBar = document.getElementById('probabilityBar');
    const factorsList = document.getElementById('factorsList');

    // Subsystem Gauges
    const gOverstrainVal = document.getElementById('gOverstrainVal');
    const gOverstrainBar = document.getElementById('gOverstrainBar');
    const gWearVal = document.getElementById('gWearVal');
    const gWearBar = document.getElementById('gWearBar');
    const gPowerVal = document.getElementById('gPowerVal');
    const gPowerBar = document.getElementById('gPowerBar');
    const gThermalVal = document.getElementById('gThermalVal');
    const gThermalBar = document.getElementById('gThermalBar');

    // Preset buttons
    const presetButtons = document.querySelectorAll('.preset-btn');

    // Modal elements
    const graphModal = document.getElementById('graphModal');
    const openGraphModalBtn = document.getElementById('openGraphModalBtn');
    const closeGraphModalBtn = document.getElementById('closeGraphModalBtn');
    const modalTabBtns = document.querySelectorAll('.modal-tab-btn');
    const modalGraphImg = document.getElementById('modalGraphImg');

    if (openGraphModalBtn && graphModal) {
        openGraphModalBtn.addEventListener('click', () => {
            graphModal.style.display = 'flex';
        });
    }

    if (closeGraphModalBtn && graphModal) {
        closeGraphModalBtn.addEventListener('click', () => {
            graphModal.style.display = 'none';
        });
    }

    if (graphModal) {
        graphModal.addEventListener('click', (e) => {
            if (e.target === graphModal) {
                graphModal.style.display = 'none';
            }
        });
    }

    modalTabBtns.forEach(tab => {
        tab.addEventListener('click', () => {
            modalTabBtns.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const imgSrc = tab.getAttribute('data-img');
            modalGraphImg.src = imgSrc;
        });
    });

    // Preset selection handler
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const presetKey = btn.getAttribute('data-preset');
            if (typeof PRESET_DATA !== 'undefined' && PRESET_DATA[presetKey]) {
                const data = PRESET_DATA[presetKey];
                
                presetButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                document.getElementById('type').value = data.type;
                document.getElementById('air_temperature').value = data.air_temperature;
                document.getElementById('process_temperature').value = data.process_temperature;
                document.getElementById('rotational_speed').value = data.rotational_speed;
                document.getElementById('torque').value = data.torque;
                document.getElementById('tool_wear').value = data.tool_wear;

                submitPrediction();
            }
        });
    });

    // Form submission
    predictionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        presetButtons.forEach(b => b.classList.remove('active'));
        submitPrediction();
    });

    async function submitPrediction() {
        formError.style.display = 'none';
        formError.textContent = '';
        btnSpinner.style.display = 'inline-block';
        predictBtn.disabled = true;

        const payload = {
            type: document.getElementById('type').value,
            air_temperature: parseFloat(document.getElementById('air_temperature').value),
            process_temperature: parseFloat(document.getElementById('process_temperature').value),
            rotational_speed: parseFloat(document.getElementById('rotational_speed').value),
            torque: parseFloat(document.getElementById('torque').value),
            tool_wear: parseFloat(document.getElementById('tool_wear').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const json = await response.json();

            if (!response.ok || !json.success) {
                throw new Error(json.error || 'Failed to generate prediction');
            }

            updateDashboard(json.data);
        } catch (err) {
            formError.textContent = err.message;
            formError.style.display = 'block';
        } finally {
            btnSpinner.style.display = 'none';
            predictBtn.disabled = false;
        }
    }

    function updateDashboard(data) {
        const isFailure = data.prediction === 1;

        // 1. Update Status Banner Style and Text
        if (isFailure) {
            resultBanner.className = 'result-banner status-failure';
            statusTitle.textContent = 'Failure Risk Detected';
            statusBadgeIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2.5" fill="none">
                    <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
            `;
        } else {
            resultBanner.className = 'result-banner status-normal';
            statusTitle.textContent = 'Normal Operation';
            statusBadgeIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2.5" fill="none">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            `;
        }

        // 2. Update Probability
        const prob = data.failure_probability;
        probabilityValue.textContent = `${prob.toFixed(1)}%`;
        probabilityBar.style.width = `${Math.max(prob, 2)}%`;

        // 3. Update Diagnostic Risk Factors
        factorsList.innerHTML = '';
        if (data.risk_factors && data.risk_factors.length > 0) {
            data.risk_factors.forEach(factor => {
                const li = document.createElement('li');
                li.textContent = factor;
                factorsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = isFailure 
                ? 'Operating parameters indicate elevated probability of mechanical failure.' 
                : 'All operating parameters are safely within standard machine tolerances.';
            factorsList.appendChild(li);
        }

        // 4. Update Subsystem Stress Gauges
        if (data.subsystem_gauges) {
            const g = data.subsystem_gauges;
            
            gOverstrainVal.textContent = `${g.overstrain_stress}%`;
            gOverstrainBar.style.width = `${Math.min(100, Math.max(2, g.overstrain_stress))}%`;
            gOverstrainBar.style.background = g.overstrain_stress > 70 ? '#ef4444' : (g.overstrain_stress > 40 ? '#f59e0b' : '#10b981');

            gWearVal.textContent = `${g.tool_wear_expended}%`;
            gWearBar.style.width = `${Math.min(100, Math.max(2, g.tool_wear_expended))}%`;
            gWearBar.style.background = g.tool_wear_expended > 75 ? '#ef4444' : (g.tool_wear_expended > 50 ? '#f59e0b' : '#10b981');

            gPowerVal.textContent = `${g.power_stress}%`;
            gPowerBar.style.width = `${Math.min(100, Math.max(2, g.power_stress))}%`;
            gPowerBar.style.background = g.power_stress > 70 ? '#ef4444' : (g.power_stress > 40 ? '#f59e0b' : '#10b981');

            gThermalVal.textContent = `${g.thermal_stress}%`;
            gThermalBar.style.width = `${Math.min(100, Math.max(2, g.thermal_stress))}%`;
            gThermalBar.style.background = g.thermal_stress > 70 ? '#ef4444' : (g.thermal_stress > 40 ? '#f59e0b' : '#10b981');
        }
    }
});
