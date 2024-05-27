document.getElementById('uploadForm').onsubmit = async function(event) {
    event.preventDefault();
    const formData = new FormData();
    const file = document.getElementById('audioFile').files[0];
    const mode = document.getElementById('mode').value;
    formData.append('file', file);
    formData.append('mode', mode);

    console.log("Sending request...");

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        console.log("Request sent. Awaiting response...");

        const result = await response.json();
        console.log("Response received:", result);

        // Clear previous results
        document.getElementById('result').innerText = "";
        const tableBody = document.getElementById('data-table').querySelector('tbody');
        tableBody.innerHTML = ""; // Clear previous table rows

        const words = result.text;
        const fp = result.FP;
        const pw = result.PW;
        const rp = result.RP;
        const rs = result.RS;
        const rv = result.RV;

        words.forEach((word, index) => {
            const row = document.createElement('tr');

            const wordCell = document.createElement('td');
            wordCell.textContent = word;
            row.appendChild(wordCell);

            const fpCell = document.createElement('td');
            fpCell.textContent = fp[index];
            row.appendChild(fpCell);

            const pwCell = document.createElement('td');
            pwCell.textContent = pw[index];
            row.appendChild(pwCell);

            const rpCell = document.createElement('td');
            rpCell.textContent = rp[index];
            row.appendChild(rpCell);

            const rsCell = document.createElement('td');
            rsCell.textContent = rs[index];
            row.appendChild(rsCell);

            const rvCell = document.createElement('td');
            rvCell.textContent = rv[index];
            row.appendChild(rvCell);

            tableBody.appendChild(row);
        });
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('result').innerText = "Error: " + error.message;
    }
};
