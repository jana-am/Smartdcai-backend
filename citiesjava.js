const BACKEND_URL = "https://dcai-backend.onrender.com";

// 1. THIS IS THE LIST THAT FILLS THE MENU
const cities = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Phoenix, AZ",
    "Las Vegas, NV", "Denton, TX", "Billings, MT", "San Diego, CA", 
    "Kodiak, AK", "Aspen, CO", "Dodge City, KS", "San Francisco, CA", 
    "Greensburg, KS", "Honolulu, HI", "Albuquerque, NM", "San Jose, CA", 
    "Tucson, AZ", "Reno, NV", "El Paso, TX", "Salt Lake City, UT"
];

// 2. THIS FUNCTION PUTS THE CITIES INTO THE DROPDOWN
function initPage() {
    const dropdown = document.getElementById('city-dropdown');
    if (!dropdown) return;

    dropdown.innerHTML = '<option value="" disabled selected>Select a city...</option>';
    
    cities.forEach(city => {
        const option = document.createElement('option');
        option.value = city;
        option.textContent = city;
        dropdown.appendChild(option);
    });

    // Update file name text when a user selects a file
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileNameDisplay.textContent = "Selected: " + e.target.files[0].name;
                fileNameDisplay.style.color = "#3ea174";
            }
        });
    }
}

// 3. ANALYZE PRE-TRAINED CITIES
async function handleCitySelect() {
    const dropdown = document.getElementById('city-dropdown');
    const selectedCity = dropdown.value;
    const token = localStorage.getItem('access_token');

    if (!selectedCity) return alert("Please select a city!");
    if (!token) return alert("Please Login first!");

    document.body.style.cursor = 'wait';
    try {
        const response = await fetch(`${BACKEND_URL}/predict/${encodeURIComponent(selectedCity)}`, {
            method: 'GET',
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        localStorage.setItem('lastPrediction', JSON.stringify(data));
        localStorage.setItem('selectedCity', selectedCity);
        window.location.href = "result.html";
    } catch (err) {
        alert("Error connecting to server.");
    } finally {
        document.body.style.cursor = 'default';
    }
}

// 4. ANALYZE NEW UPLOADED FILES
async function handleUploadAndAnalyze() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    const token = localStorage.getItem('access_token');

    if (!file) return alert("Please select a CSV file first!");
    if (!token) return alert("Please Login first!");

    document.body.style.cursor = 'wait';
    const formData = new FormData();
    formData.append('files', file);

    try {
        const response = await fetch(`${BACKEND_URL}/predict/upload`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` },
            body: formData
        });
        const data = await response.json();
        localStorage.setItem('lastPrediction', JSON.stringify(data));
        localStorage.setItem('selectedCity', "Upload: " + file.name);
        window.location.href = "result.html";
    } catch (err) {
        alert("Upload failed.");
    } finally {
        document.body.style.cursor = 'default';
    }
}

// IMPORTANT: This tells the browser to run the "initPage" function as soon as the page opens
window.addEventListener('DOMContentLoaded', initPage);