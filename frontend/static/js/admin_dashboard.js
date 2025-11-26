// Initialize data containers and charts
let predictionData = [];
let donationData = [];
let predictionChart = null;
let donationChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async function() {
    initializeCharts();
    fetchDonations();
    fetchPredictions();

    // Add event listeners for chart type changes
    document.getElementById('predictionChartType').addEventListener('change', updatePredictionChart);
});

// Initialize charts with default settings
function initializeCharts() {
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
    Chart.defaults.color = '#000000';
    Chart.defaults.plugins.legend.labels.color = '#000000';
}

// Fetch predictions data
function fetchPredictions(page = 1) {
    const loader = document.getElementById('predictionChartLoader');
    loader.style.display = 'flex';

    fetch(`/api/get_predictions?page=${page}&per_page=10`)
        .then(response => response.json())
        .then(data => {
            if (data.status === "success" && data.predictions) {
                predictionData = data.predictions;
                updatePredictionChart();
                updatePredictionTable(data.predictions, data.data_source);
                setupPagination('predictionPagination', data.total_pages || 1, page, fetchPredictions);
                
                // Update stats
                updateStats(data);
            }
            loader.style.display = 'none';
        })
        .catch(error => {
            console.error('Error fetching predictions:', error);
            loader.style.display = 'none';
        });
}

// Fetch donations data
function fetchDonations(page = 1) {
    const loader = document.getElementById('donationChartLoader');
    loader.style.display = 'flex';

    fetch(`/api/get_donations?page=${page}&per_page=10`)
        .then(response => response.json())
        .then(data => {
            if (data.status === "success" && data.donations) {
                donationData = data.donations;
                updateDonationTable(data.donations, data.data_source);
                setupPagination('donationPagination', data.total_pages || 1, page, fetchDonations);
                updateStats(data);
            }
            loader.style.display = 'none';
        })
        .catch(error => {
            console.error('Error fetching donations:', error);
            loader.style.display = 'none';
        });
}

function getChartOptions(type, isDonations = false) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    color: '#000000',
                    font: {
                        family: "'Plus Jakarta Sans', sans-serif",
                        size: 12
                    },
                    padding: 20
                }
            }
        },
        scales: type !== 'pie' && type !== 'doughnut' ? {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                },
                ticks: {
                    color: '#000000',
                    font: {
                        family: "'Plus Jakarta Sans', sans-serif"
                    },
                    callback: value => isDonations ? '₹' + value.toLocaleString() : value
                }
            },
            x: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                },
                ticks: {
                    color: '#000000',
                    font: {
                        family: "'Plus Jakarta Sans', sans-serif"
                    }
                }
            }
        } : undefined
    };
}

// Update prediction chart
function updatePredictionChart() {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    const chartType = document.getElementById('predictionChartType').value;
    
    const correctCount = predictionData.filter(p => p.is_correct).length;
    const incorrectCount = predictionData.length - correctCount;

    if (predictionChart) {
        predictionChart.destroy();
    }

    const config = {
        type: chartType,
        data: {
            labels: ['Correct Predictions', 'Incorrect Predictions'],
            datasets: [{
                label: 'Prediction Accuracy',
                data: [correctCount, incorrectCount],
                backgroundColor: [
                    '#43a047',  // Success green
                    '#e53935'   // Danger red
                ],
                borderColor: [
                    '#43a047',
                    '#e53935'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#000000',
                        font: {
                            family: "'Plus Jakarta Sans', sans-serif",
                            size: 12
                        },
                        padding: 20
                    }
                }
            },
            scales: chartType === 'line' || chartType === 'bar' ? {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        color: '#000000',
                        font: {
                            family: "'Plus Jakarta Sans', sans-serif"
                        }
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        color: '#000000',
                        font: {
                            family: "'Plus Jakarta Sans', sans-serif"
                        }
                    }
                }
            } : {}
        }
    };

    predictionChart = new Chart(ctx, config);
}

// Update prediction table
function updatePredictionTable(predictions, dataSource) {
    const tableBody = document.getElementById('predictionTableBody');
    tableBody.innerHTML = predictions.map((item, index) => `
        <tr>
            <td>${index + 1}</td>
            <td>${item.owner_name}</td>
            <td>${item.pet_name}</td>
            <td>${item.disease}</td>
            <td>
                <span class="badge badge-${item.is_correct ? 'success' : 'danger'}">
                    <i class="fas fa-${item.is_correct ? 'check' : 'times'}"></i>
                    ${item.is_correct ? 'Correct' : 'Incorrect'}
                </span>
            </td>
            <td>${dataSource === 'local' && !item.is_correct ? item.correct_label : '-'}</td>
        </tr>
    `).join('');
}

// Update donation table
function updateDonationTable(donations, dataSource) {
    const tableBody = document.getElementById('donationTableBody');
    tableBody.innerHTML = donations.map((item, index) => `
        <tr>
            <td>${index + 1}</td>
            <td>${item.donor_name}</td>
            <td>₹${parseFloat(item.amount_inr).toLocaleString()}</td>
            <td>${new Date(item.date).toLocaleDateString()}</td>
            <td>
                <button class="btn btn-sm btn-success" onclick="downloadInvoice('${item.transaction_id}')">
                    <i class="fas fa-download"></i> Invoice
                </button>
            </td>
        </tr>
    `).join('');
}

// Setup pagination
function setupPagination(containerId, totalPages, currentPage, callback) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    for (let i = 1; i <= totalPages; i++) {
        const li = document.createElement('li');
        li.className = `page-item ${i === currentPage ? 'active' : ''}`;
        
        const a = document.createElement('a');
        a.className = 'page-link';
        a.href = '#';
        a.textContent = i;
        
        a.addEventListener('click', (e) => {
            e.preventDefault();
            callback(i);
        });
        
        li.appendChild(a);
        container.appendChild(li);
    }
}

// Update statistics
function updateStats(data) {
    document.getElementById('totalPredictions').textContent = data.total_predictions || 0;
    document.getElementById('accuratePredictions').textContent = data.accurate_predictions || 0;
    document.getElementById('totalDonations').textContent = `₹${(data.grand_total_donations || 0).toLocaleString()}`;
}

// Download invoice
function downloadInvoice(transactionId) {
    window.location.href = `/static/invoices/invoice_${transactionId}.pdf`;
}