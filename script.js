// app.js

// Function to get stock data
async function getStockData(ticker, days) {
    const end_date = new Date();
    const start_date = new Date(end_date.getTime() - days * 24 * 60 * 60 * 1000);

    const response = await fetch(`/api/stock-data?ticker=${ticker}&start_date=${start_date.toISOString()}&end_date=${end_date.toISOString()}`);
    const data = await response.json();
    return data;
}

// Function to calculate RSI
function calculateRSI(data, periods = 14) {
    const delta = data.map((_, i) => i > 0 ? data[i] - data[i - 1] : 0);
    const gain = delta.map(d => d > 0 ? d : 0).reduce((a, b) => a + b, 0) / periods;
    const loss = delta.map(d => d < 0 ? -d : 0).reduce((a, b) => a + b, 0) / periods;
    const rs = gain / loss;
    return 100 - (100 / (1 + rs));
}

// Function to prepare data
function prepareData(data) {
    data.Target = data.Close.slice(1);
    data.MA7 = data.Close.rolling(7).mean();
    data.MA21 = data.Close.rolling(21).mean();
    data.RSI = data.Close.map(calculateRSI);
    data.MACD = data.Close.ewm(span=12).mean() - data.Close.ewm(span=26).mean();
    data.SignalLine = data.MACD.ewm(span=9).mean();
    data.ATR = calculateATR(data);
    return data.dropna();
}

// Function to calculate ATR
function calculateATR(data, period = 14) {
    const highLow = data.High - data.Low;
    const highClose = data.High.map((h, i) => Math.abs(h - (data.Close[i] || 0)));
    const lowClose = data.Low.map((l, i) => Math.abs(l - (data.Close[i] || 0)));
    const ranges = highLow.concat(highClose, lowClose);
    const trueRange = ranges.reduce((a, b) => Math.max(a, b), 0);
    return trueRange.rolling(period).mean();
}

// Function to train and predict
async function trainAndPredict(X_train, X_test, y_train, y_test) {
    const results = {};

    const models = {
        'Random Forest': new RandomForestRegressor(100, 42),
        'Linear Regression': new LinearRegression(),
        'Support Vector Regression': new SVR({ kernel: 'rbf' }),
        'Decision Tree': new DecisionTreeRegressor(42),
        'K-Nearest Neighbors': new KNeighborsRegressor(5)
    };

    for (const [name, model] of Object.entries(models)) {
        await model.fit(X_train, y_train);
        const prediction = await model.predict([X_test.slice(-1)[0]]);
        const accuracy = await model.score(X_test, y_test);
        results[name] = { prediction: prediction[0], accuracy };
    }

    return results;
}

// Event listener for the "Analyze and Predict" button
document.getElementById('analyze-button').addEventListener('click', async () => {
    const ticker = document.getElementById('ticker').value;
    const days = document.getElementById('days').value;

    const data = await getStockData(ticker, days);
    const X_train, X_test, y_train, y_test = prepareData(data);
    const predictions = await trainAndPredict(X_train, X_test, y_train, y_test);

    // Display results
    document.getElementById('results').style.display = 'block';
    document.getElementById('current-price').innerHTML = `
        <div class="metric">
            <h3>Current Price</h3>
            <p>$${data.Close.slice(-1)[0].toFixed(2)}</p>
        </div>
    `;

    document.getElementById('model-predictions').innerHTML = `
        <h3>Model Predictions</h3>
        ${Object.entries(predictions).map(([model, result]) => `
            <div class="metric">
                <h4>${model}</h4>
                <p>$${result.prediction.toFixed(2)} (${((result.prediction - data.Close.slice(-1)[0]) / data.Close.slice(-1)[0] * 100).toFixed(2)}%)</p>
                <p>Accuracy: ${(result.accuracy * 100).toFixed(2)}%</p>
            </div>
        `).join('')}
    `;

    document.getElementById('best-model').innerHTML = `
        <div class="metric">
            <h3>Best Performing Model</h3>
            ${Object.entries(predictions).reduce((best, [model, result]) => best.accuracy > result.accuracy ? best : { model, ...result }, { model: '', prediction: 0, accuracy: 0 }).model}
            (Accuracy: ${(Object.entries(predictions).reduce((best, [model, result]) => best.accuracy > result.accuracy ? best : { model, ...result }, { model: '', prediction: 0, accuracy: 0 }).accuracy * 100).toFixed(2)}%)
        </div>
    `;

    // Plot historical data with candlestick chart
    Plotly.newPlot('historical-chart', [{
        type: 'candlestick',
        x: data.index,
        open: data.Open,
        high: data.High,
        low: data.Low,
        close: data.Close
    }], {
        title: `${ticker} Stock Price`,
        yaxis_title: 'Price',
        xaxis_title: 'Date'
    });

    // Plot volume
    Plotly.newPlot('volume-chart', [{
        type: 'bar',
        x: data.index,
        y: data.Volume
    }], {
        title: `${ticker} Trading Volume`
    });

    // Plot RSI
    Plotly.newPlot('rsi-chart', [{
        type: 'line',
        x: data.index,
        y: data.RSI
    }], {
        title: 'Relative Strength Index (RSI)',
        shapes: [
            {type: 'line', x0: 0, x1: 1, y0: 70, y1: 70, line: {dash: 'dash', color: 'red'}},
            {type: 'line', x0: 0, x1: 1, y0: 30, y1: 30, line: {dash: 'dash', color: 'green'}}
        ]
    });

    // Plot MACD
    Plotly.newPlot('macd-chart', [{
        type: 'line',
        x: data.index,
        y: data.MACD
    }, {
        type: 'line',
        x: data.index,
        y: data.SignalLine
    }], {
        title: 'MACD'
    });
});

// Machine learning models
class RandomForestRegressor {
    constructor(n_estimators, random_state) {
        this.n_estimators = n_estimators;
        this.random_state = random_state;
    }

    async fit(X, y) {
        // Implement random forest regression training logic
    }

    async predict(X) {
        // Implement random forest regression prediction logic
    }

    async score(X, y) {
        // Implement random forest regression model evaluation logic
    }
}

class LinearRegression {
    async fit(X, y) {
        // Implement linear regression training logic
    }

    async predict(X) {
        // Implement linear regression prediction logic
    }

    async score(X, y) {
        // Implement linear regression model evaluation logic
    }
}

class SVR {
    constructor(options) {
        this.options = options;
    }

    async fit(X, y) {
        // Implement support vector regression training logic
    }

    async predict(X) {
        // Implement support vector regression prediction logic
    }

    async score(X, y) {
        // Implement support vector regression model evaluation logic
    }
}

class DecisionTreeRegressor {
    constructor(random_state) {
        this.random_state = random_state;
    }

    async fit(X, y) {
        // Implement decision tree regression training logic
    }

    async predict(X) {
        // Implement decision tree regression prediction logic
    }

    async score(X, y) {
        // Implement decision tree regression model evaluation logic
    }
}

class KNeighborsRegressor {
    constructor(n_neighbors) {
        this.n_neighbors = n_neighbors;
    }

    async fit(X, y) {
        // Implement k-nearest neighbors regression training logic
    }

    async predict(X) {
        // Implement k-nearest neighbors regression prediction logic
    }

    async score(X, y) {
        // Implement k-nearest neighbors regression model evaluation logic
    }
}