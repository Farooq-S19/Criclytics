// --- START OF FILE static/js/analysis.js ---

document.addEventListener('DOMContentLoaded', function() {
    // --- Common Elements ---
    let charts = {};

    // --- Dynamic Population of Dropdowns ---
    function populateDropdown(selectElement, items, defaultOptionText) {
        if (!selectElement) return;
        
        // Clear existing options except the first disabled one
        selectElement.innerHTML = `<option value="" disabled selected>${defaultOptionText}</option>`;
        
        items.forEach(item => {
            const option = document.createElement('option');
            option.value = item;
            option.textContent = item;
            selectElement.appendChild(option);
        });
    }

    // Fetch initial data for dropdowns
    fetch('/get_initial_data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const playerSelect = document.getElementById('player-select');
            const team1Select = document.getElementById('team1-select');
            const team2Select = document.getElementById('team2-select');
            
            if (playerSelect) {
                populateDropdown(playerSelect, data.players, 'Select a Player...');
            }
            if (team1Select) {
                populateDropdown(team1Select, data.teams, 'Select Team A...');
            }
            if (team2Select) {
                populateDropdown(team2Select, data.teams, 'Select Team B...');
            }
        })
        .catch(error => {
            console.error("Fatal Error: Could not load initial player/team data.", error);
            const playerSelect = document.getElementById('player-select');
            if(playerSelect) playerSelect.innerHTML = '<option value="" disabled selected>Error loading players</option>';
            const team1Select = document.getElementById('team1-select');
            if(team1Select) team1Select.innerHTML = '<option value="" disabled selected>Error loading teams</option>';
            const team2Select = document.getElementById('team2-select');
            if(team2Select) team2Select.innerHTML = '<option value="" disabled selected>Error loading teams</option>';
            alert("Error: Could not load initial player and team data. Please check the server.");
        });


    // --- Page-specific Initializations ---
    const playerSelect = document.getElementById('player-select');
    if (playerSelect) {
        initializePlayerAnalysis();
    }
    
    const predictTeamBtn = document.getElementById('predict-team-btn');
    if (predictTeamBtn) {
        initializeTeamPrediction();
    }

    // --- PLAYER ANALYSIS PAGE LOGIC ---
    function initializePlayerAnalysis() {
        playerSelect.addEventListener('change', function() {
            const playerName = this.value;
            if (!playerName) return;

            document.getElementById('loader').style.display = 'block';
            document.getElementById('player-dashboard').style.display = 'none';

            fetch('/get_player_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ player_name: playerName }),
            })
            .then(response => {
                if (!response.ok) {
                    // If response is not OK (e.g., 500, 404), throw an error
                    // Read response as text to potentially get detailed error from backend
                    return response.text().then(text => { throw new Error(text || response.statusText); });
                }
                return response.json();
            })
            .then(data => {
                document.getElementById('loader').style.display = 'none';
                document.getElementById('player-dashboard').style.display = 'grid';
                updatePlayerUI(data);
            })
            .catch(error => {
                console.error("Error fetching player data:", error);
                document.getElementById('loader').style.display = 'none';
                alert("An error occurred while fetching player data. Details: " + error.message);
                document.getElementById('player-dashboard').style.display = 'none'; // Ensure dashboard stays hidden
            });
        });
    }
    
    function updatePlayerUI(data) {
        // CRITICAL FIX: Robustness checks for expected data structure
        if (!data || !data.profile || !data.prediction || !data.charts) {
            console.error("Received invalid data structure or backend error:", data);
            alert("Failed to load player data. Data structure invalid. Check server logs.");
            document.getElementById('player-dashboard').style.display = 'none'; // Keep dashboard hidden
            return; // Exit the function if data is invalid
        }

        // Update Profile Card
        const profileCard = document.getElementById('player-profile-card');
        if (profileCard) {
            profileCard.innerHTML = `
                <div class="profile-header">
                    <h2>${data.profile.name || 'N/A'}</h2>
                    <span>${data.profile.team || 'N/A'}</span>
                </div>
                <div class="profile-details">
                    <div><strong>Role:</strong> ${data.profile.role || 'N/A'}</div>
                    <div><strong>Nationality:</strong> ${data.profile.nationality || 'N/A'}</div>
                    <div><strong>Age:</strong> ${data.profile.age || 'N/A'}</div>
                </div>
                <div class="profile-summary">
                    <p>${data.summary || 'No summary available.'}</p>
                </div>
            `;
        }

        // Update Prediction Box
        const predictionBox = document.getElementById('ai-prediction-box');
        if (predictionBox) {
            predictionBox.innerHTML = `
                <h3>AI Prediction</h3>
                <div class="prediction-main">
                    <span class="score">${data.prediction.score_range || 'N/A'}</span>
                </div>
                <div class="prediction-confidence">
                    <span>Confidence:</span>
                    <div class="confidence-bar"><div style="width: ${data.prediction.confidence || '0%'};"></div></div>
                    <span>${data.prediction.confidence || '0%'}</span>
                </div>
            `;
        }

        // Update Charts - Ensure chart data is valid before passing
        if (data.charts.recent_performance) {
            createOrUpdateChart('bar-chart', 'bar', data.charts.recent_performance, 'Runs per Match');
        } else {
            console.warn("Missing recent_performance_data for bar chart.");
        }
        
        if (data.charts.performance_breakdown) {
            createOrUpdateChart('pie-chart', 'pie', data.charts.performance_breakdown, 'Batting Contribution');
        } else {
            console.warn("Missing performance_breakdown_data for pie chart.");
        }

        if (data.charts.skills_radar) {
            createOrUpdateChart('radar-chart', 'radar', data.charts.skills_radar, 'Player Skills');
        } else {
            console.warn("Missing skills_radar_data for radar chart.");
        }
    }

    // --- TEAM PREDICTION PAGE LOGIC ---
    function initializeTeamPrediction() {
        const team1Select = document.getElementById('team1-select');
        const team2Select = document.getElementById('team2-select');
        
        predictTeamBtn.addEventListener('click', function() {
            const team1 = team1Select.value;
            const team2 = team2Select.value;
            
            if (!team1 || !team2 || team1 === team2) {
                alert("Please select two different teams.");
                return;
            }

            document.getElementById('team-loader').style.display = 'block';
            document.getElementById('team-prediction-result').style.display = 'none';

            fetch('/get_team_prediction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ team1: team1, team2: team2 }),
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => { throw new Error(text || response.statusText); });
                }
                return response.json();
            })
            .then(data => {
                document.getElementById('team-loader').style.display = 'none';
                document.getElementById('team-prediction-result').style.display = 'grid';
                updateTeamUI(data, team1, team2);
            })
            .catch(error => {
                console.error("Error fetching team prediction:", error);
                document.getElementById('team-loader').style.display = 'none';
                alert("An error occurred while fetching the prediction. Details: " + error.message);
                document.getElementById('team-prediction-result').style.display = 'none'; // Ensure hidden
            });
        });
    }

    function updateTeamUI(data, team1, team2) {
        // Robustness check for team prediction data
        if (!data || !data.prediction || !data.team1_players || !data.team2_players || !data.team1_summary || !data.team2_summary) {
            console.error("Received invalid data structure for team prediction:", data);
            alert("Failed to load team prediction data. Data structure invalid. Check server logs.");
            document.getElementById('team-prediction-result').style.display = 'none';
            return;
        }

        document.getElementById('winner-name').textContent = data.prediction.winner || 'N/A';
        document.getElementById('win-probability-text').textContent = data.prediction.confidence || '0%';
        document.getElementById('prediction-reasoning').textContent = data.prediction.reason || 'No analysis available.';

        // Update probability bars
        document.getElementById('team1-prob-bar').style.width = (data.prediction.prob1 || 0) + '%';
        document.getElementById('team1-prob-text').textContent = `${team1}: ${data.prediction.prob1 || '0'}%`;
        document.getElementById('team2-prob-bar').style.width = (data.prediction.prob2 || 0) + '%';
        document.getElementById('team2-prob-text').textContent = `${team2}: ${data.prediction.prob2 || '0'}%`;
        
        // Populate team summary tables
        const team1SummaryCard = document.getElementById('team1-summary');
        const team2SummaryCard = document.getElementById('team2-summary');

        if (team1SummaryCard && data.team1_summary) {
            let summaryHtml1 = `<h4>${team1} Stats</h4><table><tbody>`;
            for (const [key, value] of Object.entries(data.team1_summary)) {
                summaryHtml1 += `<tr><th>${key}:</th><td>${value || 'N/A'}</td></tr>`;
            }
            summaryHtml1 += `</tbody></table>`;
            team1SummaryCard.innerHTML = summaryHtml1;
        } else {
            team1SummaryCard.innerHTML = `<h4>${team1} Stats</h4><p>No summary data.</p>`;
        }

        if (team2SummaryCard && data.team2_summary) {
            let summaryHtml2 = `<h4>${team2} Stats</h4><table><tbody>`;
            for (const [key, value] of Object.entries(data.team2_summary)) {
                summaryHtml2 += `<tr><th>${key}:</th><td>${value || 'N/A'}</td></tr>`;
            }
            summaryHtml2 += `</tbody></table>`;
            team2SummaryCard.innerHTML = summaryHtml2;
        } else {
            team2SummaryCard.innerHTML = `<h4>${team2} Stats</h4><p>No summary data.</p>`;
        }

        // Populate player lists
        const team1List = document.getElementById('team1-players');
        const team2List = document.getElementById('team2-players');
        team1List.innerHTML = `<h4>${team1}</h4><ul>${data.team1_players.map(p => `<li>${p || 'Unknown Player'}</li>`).join('')}</ul>`;
        team2List.innerHTML = `<h4>${team2}</h4><ul>${data.team2_players.map(p => `<li>${p || 'Unknown Player'}</li>`).join('')}</ul>`;
    }

    // --- Chart.js Helper Function (ENHANCED) ---
    function createOrUpdateChart(chartId, type, chartData, label) {
        const chartElement = document.getElementById(chartId);
        if (!chartElement) return;

        const ctx = chartElement.getContext('2d');
        if (charts[chartId]) {
            charts[chartId].destroy();
        }

        // Determine theme-dependent colors dynamically for better visibility
        const isDarkTheme = document.body.getAttribute('data-theme') === 'dark';
        const dynamicGridLineColor = isDarkTheme ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'; // White for dark, black for light
        const dynamicFontColor = getComputedStyle(document.documentElement).getPropertyValue('--font-color').trim();

        // Chart.js Dataset configuration
        const datasetsConfig = [{
            label: label,
            data: chartData.data,
            // Main fill color (more opaque for light theme for better contrast)
            backgroundColor: (type === 'radar') ? (isDarkTheme ? 'rgba(59, 130, 246, 0.2)' : 'rgba(59, 130, 246, 0.4)') : ['#3b82f6', '#10b981', '#ef4444', '#f97316', '#8b5cf6'],
            // Border color of the radar shape
            borderColor: (type === 'radar') ? '#3b82f6' : 'var(--bg-secondary)',
            borderWidth: (type === 'radar') ? 2 : 1, // Thicker border for radar chart

            // Radar Chart Specific Point Styling
            pointBackgroundColor: (type === 'radar') ? (isDarkTheme ? '#fff' : '#3b82f6') : undefined, // White points for dark, blue for light
            pointBorderColor: (type === 'radar') ? (isDarkTheme ? '#3b82f6' : '#fff') : undefined, // Blue border for dark, white for light
            pointRadius: (type === 'radar') ? 4 : undefined,
            pointHoverRadius: (type === 'radar') ? 6 : undefined,
            pointHitRadius: (type === 'radar') ? 10 : undefined,
            pointBorderWidth: (type === 'radar') ? 2 : undefined,
        }];

        // Chart.js Options configuration
        let chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: (type === 'radar') ? false : true, // Hide legend for radar chart if only one dataset
                    labels: { color: dynamicFontColor } // Legend labels adjust to theme
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            // For pie chart, show raw score (number of runs)
                            if (type === 'pie') {
                                // context.raw already holds the numeric value you want (e.g., 38.8)
                                return context.label + ': ' + context.raw + ' runs'; // Changed to raw score + ' runs'
                            }
                            return context.dataset.label + ': ' + context.raw;
                        }
                    }
                }
            },
            scales: {} // Initialize scales object
        };

        if (type === 'bar') {
            chartOptions.scales = {
                y: {
                    beginAtZero: true,
                    ticks: { color: dynamicFontColor }, // Y-axis ticks adjust to theme
                    grid: { color: dynamicGridLineColor } // Y-axis grid lines adjust to theme
                },
                x: {
                    ticks: { color: dynamicFontColor }, // X-axis ticks adjust to theme
                    grid: { display: false } // X-axis grid lines usually hidden for bar charts
                }
            };
        } else if (type === 'radar') {
            chartOptions.scales.r = { // Radial axis for radar charts
                angleLines: {
                    color: dynamicGridLineColor, // Angle lines adjust to theme
                    lineWidth: 1
                },
                grid: {
                    color: dynamicGridLineColor, // Concentric grid lines adjust to theme
                    lineWidth: 1
                },
                ticks: {
                    backdropColor: 'transparent', // No background behind tick labels
                    color: dynamicFontColor, // Tick labels adjust to theme
                    font: { size: 10 },
                    stepSize: 20 // Assuming skills are 0-100, adjust if needed
                },
                pointLabels: { // Labels like "Consistency", "Strike Rate"
                    color: dynamicFontColor, // Point labels adjust to theme
                    font: {
                        size: 12,
                        weight: 'bold'
                    }
                },
                min: 0, // Set minimum value of the scale
                max: 100 // Set maximum value of the scale (adjust based on your actual skill rating range)
            };
        }

        charts[chartId] = new Chart(ctx, {
            type: type,
            data: {
                labels: chartData.labels,
                datasets: datasetsConfig // Use the defined datasetsConfig
            },
            options: chartOptions // Use the defined chartOptions
        });
    }
});
// --- END OF FILE static/js/analysis.js ---