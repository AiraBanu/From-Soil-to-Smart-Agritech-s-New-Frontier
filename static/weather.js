const API_KEY = '0771bb71db6879ef89879e6b552df989';
const locationEl = document.getElementById('location');
const forecastContainer = document.getElementById('forecast-container');
const currentWeather = document.getElementById('current-weather');

// Function to get weather icon based on weather code
function getWeatherIcon(weatherCode) {
    return `https://openweathermap.org/img/wn/${weatherCode}@2x.png`;
}

// Function to get location name from coordinates
async function getLocationName(lat, lon) {
    try {
        const response = await fetch(`https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${API_KEY}`);
        const data = await response.json();
        return data[0].name + ', ' + data[0].country;
    } catch (error) {
        console.error('Error getting location name:', error);
        return 'Location not found';
    }
}

// //
// async function getLocationName(lat,lon) {

//     return "Nagole" ;
// }

// Function to get weather data
async function getWeatherData() {
    try {
        const position = await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject);
        });

        const { latitude: lat, longitude: lon } = position.coords;
        
        // Get location name
        const locationName = await getLocationName(lat, lon);
        locationEl.textContent = locationName;

        // Get current weather
        const currentWeatherResponse = await fetch(
            `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${API_KEY}`
        );
        const currentWeatherData = await currentWeatherResponse.json();

        // Get 5-day forecast
        const forecastResponse = await fetch(
            `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&units=metric&appid=${API_KEY}`
        );
        const forecastData = await forecastResponse.json();

        displayWeather(currentWeatherData, forecastData);
    } catch (error) {
        console.error('Error:', error);
        locationEl.textContent = 'Error loading weather data';
    }
}

// Add this function to generate farming advice based on weather
function generateFarmingAdvice(weatherData) {
    const temp = weatherData.main.temp;
    const humidity = weatherData.main.humidity;
    const weatherDesc = weatherData.weather[0].main.toLowerCase();
    
    let advice = [];
    
    // Temperature based advice
    if (temp > 30) {
        advice.push("Consider additional irrigation due to high temperatures.");
        advice.push("Best to avoid spraying pesticides in hot conditions.");
    } else if (temp < 10) {
        advice.push("Protect sensitive crops from cold conditions.");
        advice.push("Morning frost possible - delay morning field operations.");
    }

    // Weather condition based advice
    if (weatherDesc.includes('rain')) {
        advice.push("Hold off on spraying operations.");
        advice.push("Good conditions for transplanting.");
    } else if (weatherDesc.includes('clear')) {
        advice.push("Favorable conditions for harvesting.");
        advice.push("Consider irrigation if needed.");
    }

    // Humidity based advice
    if (humidity > 80) {
        advice.push("Watch for fungal disease development.");
        advice.push("Ensure good ventilation in greenhouses.");
    }

    return advice.join('<br>• ');
}

// Function to display weather data
function displayWeather(currentData, forecastData) {
    // Display current weather
    const currentTemp = currentData.main.temp.toFixed(1);
    const currentDescription = currentData.weather[0].description;
    const currentIcon = currentData.weather[0].icon;

    currentWeather.innerHTML = `
        <div class="current-temp">
            <img src="${getWeatherIcon(currentIcon)}" alt="weather icon" class="w-icon">
            <span>${currentTemp}°C</span>
        </div>
        <div class="weather-description">${currentDescription}</div>
    `;

    // Add farming advice
    const farmingAdviceEl = document.getElementById('farming-advice');
    const advice = generateFarmingAdvice(currentData);
    farmingAdviceEl.innerHTML = `• ${advice}`;

    // Add weather warning if necessary
    const weatherWarningEl = document.getElementById('weather-warning');
    if (currentData.main.temp > 35 || currentData.main.temp < 5) {
        weatherWarningEl.innerHTML = `Extreme temperature conditions may affect crops. Take necessary precautions.`;
        document.querySelector('.weather-alert').style.display = 'block';
    } else {
        document.querySelector('.weather-alert').style.display = 'none';
    }

    // Process and display forecast
    const dailyForecasts = {};
    forecastData.list.forEach(forecast => {
        const date = new Date(forecast.dt * 1000);
        const day = date.toLocaleDateString();
        
        if (!dailyForecasts[day]) {
            dailyForecasts[day] = {
                day: new Date(forecast.dt * 1000).toLocaleDateString('en-US', { weekday: 'short' }),
                icon: forecast.weather[0].icon,
                minTemp: forecast.main.temp_min,
                maxTemp: forecast.main.temp_max
            };
        } else {
            dailyForecasts[day].minTemp = Math.min(dailyForecasts[day].minTemp, forecast.main.temp_min);
            dailyForecasts[day].maxTemp = Math.max(dailyForecasts[day].maxTemp, forecast.main.temp_max);
        }
    });

    // Create forecast HTML
    forecastContainer.innerHTML = Object.values(dailyForecasts)
        .slice(1, 6)
        .map(forecast => `
            <div class="grid-item">
                <h3>${forecast.day}</h3>
                <img src="${getWeatherIcon(forecast.icon)}" alt="weather icon" class="w-icon">
                <h4>${forecast.maxTemp.toFixed(1)}°C</h4>
                <span class="min-temp">${forecast.minTemp.toFixed(1)}°C</span>
            </div>
        `).join('');
}

// Add error handling for geolocation
function handleGeolocationError(error) {
    let errorMessage;
    switch(error.code) {
        case error.PERMISSION_DENIED:
            errorMessage = "Please enable location services to get weather information.";
            break;
        case error.POSITION_UNAVAILABLE:
            errorMessage = "Location information unavailable.";
            break;
        case error.TIMEOUT:
            errorMessage = "Location request timed out.";
            break;
        default:
            errorMessage = "An unknown error occurred.";
    }
    locationEl.textContent = errorMessage;
}

// Initialize weather data
if (navigator.geolocation) {
    getWeatherData();
} else {
    locationEl.textContent = "Geolocation is not supported by this browser.";
}

// Refresh weather data every 30 minutes
setInterval(getWeatherData, 30 * 60 * 1000);

