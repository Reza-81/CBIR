// Store the unique filename globally
let uniqueFilename = null;

// Display the image immediately after selection
document.getElementById('fileInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const queryImage = document.getElementById('queryImage');
        queryImage.src = URL.createObjectURL(file);
        queryImage.style.display = 'block';
    }
});

// Perform the search (and upload the file)
async function performSearch() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select a file first!");
        return;
    }

    try {
        // Upload the file
        const formData = new FormData();
        formData.append('file', file);
        const uploadResponse = await fetch('/upload', { method: 'POST', body: formData });
        const uploadResult = await uploadResponse.json();
        
        if (!uploadResponse.ok) {
            throw new Error(uploadResult.error);
        }

        // Store the unique filename returned by the server
        uniqueFilename = uploadResult.filename;

        // Display the uploaded image
        const queryImage = document.getElementById('queryImage');
        queryImage.src = URL.createObjectURL(file);
        queryImage.style.display = 'block';

        // Perform the search using the unique filename
        const searchResponse = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: uniqueFilename })  // Use the unique filename
        });
        const searchResult = await searchResponse.json();
        
        if (!searchResponse.ok) {
            throw new Error(searchResult.error);
        }

        // Display the results
        displayResults(searchResult.results);
    } catch (error) {
        alert(error.message);
    }
}

// Refine the search
async function refineSearch() {
    if (!uniqueFilename) {
        alert("Please perform an initial search first!");
        return;
    }

    // Collect feedback
    const relevantPaths = [];
    const nonRelevantPaths = [];
    document.querySelectorAll('.relevant-checkbox:checked').forEach(checkbox => {
        relevantPaths.push(checkbox.dataset.path);
    });
    document.querySelectorAll('.non-relevant-checkbox:checked').forEach(checkbox => {
        nonRelevantPaths.push(checkbox.dataset.path);
    });

    try {
        const response = await fetch('/refine_search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: uniqueFilename,  // Use the unique filename
                relevant: relevantPaths,
                non_relevant: nonRelevantPaths
            })
        });
        const result = await response.json();
        if (response.status === 200) {
            displayResults(result.results);
        } else {
            alert(result.error);
        }
    } catch (error) {
        alert("Error refining search: " + error.message);
    }
}

// Display results
function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    results.forEach(result => {
        const container = document.createElement('div');
        container.className = 'result-container';
        container.innerHTML = `
            <img src="${result.url}" class="result-image">
            <div class="feedback-controls">
                <label><input type="checkbox" class="relevant-checkbox" data-path="${result.path}"> Relevant</label>
                <label><input type="checkbox" class="non-relevant-checkbox" data-path="${result.path}"> Irrelevant</label>
            </div>
            <div class="result-score">Similarity: ${result.score.toFixed(4)}</div>
        `;
        resultsDiv.appendChild(container);
    });
}

// Event listeners
document.getElementById('searchButton').addEventListener('click', performSearch);
document.getElementById('refineButton').addEventListener('click', refineSearch);