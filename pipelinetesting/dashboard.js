document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('predictionChart').getContext('2d');
    var predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: JSON.parse(document.getElementById('dates-data').textContent),
            datasets: [{
                label: 'Predicted Prices',
                data: JSON.parse(document.getElementById('predictions-data').textContent),
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
        options: {
            scales: {
                x: {
                    ticks: {
                        maxRotation: 90,
                        minRotation: 90
                    }
                }
            }
        }
    });
});
