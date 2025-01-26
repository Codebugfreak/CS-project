const metaData = JSON.parse(document.getElementById("metadata-data").textContent);

function updateLayerOptions() {
    console.log("Updating layer options...");
    const model = document.getElementById("model_name").value;
    console.log("Selected model:", model);

    const layerSelect = document.getElementById("layer_name");

    // Clear existing options
    layerSelect.innerHTML = "";
    
    if (metaData[model]) {
        for (const [key, value] of Object.entries(metaData[model])) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = key.charAt(0).toUpperCase() + key.slice(1);

            // Add a unique ID based on the model and layer key
            option.id = `${model}-${key}`; 

            layerSelect.appendChild(option);
        }
    } else {
        // Handle invalid metadata cases (e.g., undefined model)
        const defaultOption = document.createElement("option");
        defaultOption.value = "";
        defaultOption.textContent = "No layers available";
        layerSelect.appendChild(defaultOption);
    }
}
