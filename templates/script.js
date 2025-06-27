const timeEl = document.getElementById('time');
const dateEl = document.getElementById('date');
const currentWeatherItemsEl = document.getElementById('current-weather-items');
const timezone = document.getElementById('time-zone');
const countryEl = document.getElementById('country');
const weatherForecastEl = document.getElementById('weather-forecast');
const currentTempEl = document.getElementById('current-temp');

const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const API_KEY ='0771bb71db6879ef89879e6b552df989';

// Update clock
setInterval(() => {
    const time = new Date();
    const month = time.getMonth();
    const date = time.getDate();
    const day = time.getDay();
    const hour = time.getHours();
    const hoursIn12HrFormat = hour >= 13 ? hour % 12 : hour
    const minutes = time.getMinutes();
    const ampm = hour >= 12 ? 'PM' : 'AM'

    timeEl.innerHTML = (hoursIn12HrFormat < 10 ? '0'+hoursIn12HrFormat : hoursIn12HrFormat) + ':' + 
                      (minutes < 10 ? '0'+minutes : minutes) + ' ' + 
                      `<span id="am-pm">${ampm}</span>`

    dateEl.innerHTML = days[day] + ', ' + date + ' ' + months[month]
}, 1000);

function getWeatherData() {
    navigator.geolocation.getCurrentPosition((success) => {
        const { latitude, longitude } = success.coords;

        // First, get the current weather
        fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${API_KEY}`)
            .then(res => res.json())
            .then(currentData => {
                // Then, get the 5-day forecast
                fetch(`https://api.openweathermap.org/data/2.5/forecast?lat=${latitude}&lon=${longitude}&units=metric&appid=${API_KEY}`)
                    .then(res => res.json())
                    .then(forecastData => {
                        showWeatherData(currentData, forecastData);
                    })
                    .catch(err => console.error('Forecast Error:', err));
            })
            .catch(err => console.error('Current Weather Error:', err));
    }, (error) => {
        console.error('Geolocation error:', error);
        alert('Please enable location services to get weather information');
    });
}

function showWeatherData(currentData, forecastData) {
    // Current weather
    const { main, wind, sys, name } = currentData;

    timezone.innerHTML = name;
    countryEl.innerHTML = `${currentData.coord.lat.toFixed(2)}°N ${currentData.coord.lon.toFixed(2)}°E`;

    currentWeatherItemsEl.innerHTML = `
        <div class="weather-item">
            <div>Temperature</div>
            <div>${main.temp.toFixed(1)}°C</div>
        </div>
        <div class="weather-item">
            <div>Humidity</div>
            <div>${main.humidity}%</div>
        </div>
        <div class="weather-item">
            <div>Pressure</div>
            <div>${main.pressure} hPa</div>
        </div>
        <div class="weather-item">
            <div>Wind Speed</div>
            <div>${wind.speed} m/s</div>
        </div>
        <div class="weather-item">
            <div>Sunrise</div>
            <div>${new Date(sys.sunrise * 1000).toLocaleTimeString('en-US')}</div>
        </div>
        <div class="weather-item">
            <div>Sunset</div>
            <div>${new Date(sys.sunset * 1000).toLocaleTimeString('en-US')}</div>
        </div>
    `;

    // Current day weather
    currentTempEl.innerHTML = `
        <img src="http://openweathermap.org/img/wn/${currentData.weather[0].icon}@4x.png" alt="weather icon" class="w-icon">
        <div class="other">
            <div class="day">Today</div>
            <div class="temp">Current - ${main.temp.toFixed(1)}°C</div>
            <div class="temp">Feels like - ${main.feels_like.toFixed(1)}°C</div>
        </div>
    `;

    // Process 5-day forecast
    let otherDayForcast = '';
    const dailyForecasts = {};

    // Group forecasts by day
    forecastData.list.forEach(forecast => {
        const date = new Date(forecast.dt * 1000);
        const day = date.toLocaleDateString();
        
        if (!dailyForecasts[day]) {
            dailyForecasts[day] = {
                date: date,
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
    Object.values(dailyForecasts).slice(1, 6).forEach(forecast => {
        otherDayForcast += `
            <div class="weather-forecast-item">
                <div class="day">${days[forecast.date.getDay()]}</div>
                <img src="http://openweathermap.org/img/wn/${forecast.icon}@2x.png" alt="weather icon" class="w-icon">
                <div class="temp">Min - ${forecast.minTemp.toFixed(1)}°C</div>
                <div class="temp">Max - ${forecast.maxTemp.toFixed(1)}°C</div>
            </div>
        `;
    });

    weatherForecastEl.innerHTML = otherDayForcast;
}

// Initial weather data fetch
getWeatherData();