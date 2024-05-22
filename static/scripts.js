document.addEventListener('DOMContentLoaded', function () {
    const ctx = document.getElementById('activityChart').getContext('2d');

    // 原始數據
    const activityDataRaw = [
        { label: '校外實習', value: 36.36 },
        { label: '各式比賽', value: 7.27 },
        { label: '打工', value: 20 },
        { label: '企業參訪', value: 10.91 },
        { label: '學校社團', value: 16.36 },
        { label: '培訓課程', value: 9.09 }
    ];

    // 按照值進行排序（降序）
    const activityDataSorted = activityDataRaw.sort((a, b) => b.value - a.value);

    // 提取標籤和數據
    const labels = activityDataSorted.map(item => item.label);
    const data = activityDataSorted.map(item => item.value);

    // 配置圖表數據
    const activityData = {
        labels: labels,
        datasets: [{
            data: data,
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.5)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)'
            ],
            hoverOffset: 4
        }]
    };

    // 創建圓餅圖
    const activityChart = new Chart(ctx, {
        type: 'pie',
        data: activityData,
        options: {
            maintainAspectRatio: false,
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        position: 'top',
                        font: {
                            size: 16 // 設置標籤字體大小為16
                        },
                        padding: 20 // 設置標籤與圖表之間的距離
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (tooltipItem) {
                            const label = activityData.labels[tooltipItem.dataIndex];
                            const value = activityData.datasets[0].data[tooltipItem.dataIndex];
                            return `${label}: ${value}%`;
                        }
                    }
                }
            },
            layout: {
                padding: {
                    left: 30,
                    right: 30,
                    top: 30,
                    bottom: 0
                }
            }
        }
    });

    document.getElementById('activityChart').addEventListener('mousemove', function(event) {
        const points = activityChart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);
        if (points.length) {
            const firstPoint = points[0];
            const label = activityChart.data.labels[firstPoint.index];
            const value = activityChart.data.datasets[firstPoint.datasetIndex].data[firstPoint.index];
            
            const tooltipEl = document.getElementById('detail');
            tooltipEl.style.display = 'block';
            tooltipEl.style.left = event.pageX + 'px';
            tooltipEl.style.top = event.pageY + 'px';
            document.getElementById('detailTitle').innerText = label;
            document.getElementById('detailIndex').innerText = `推薦指數: ${value}`;
        } else {
            document.getElementById('detail').style.display = 'none';
        }
    });

    document.getElementById('activityChart').addEventListener('mouseout', function() {
        document.getElementById('detail').style.display = 'none';
    });
});
