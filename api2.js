document.getElementById("prediction-form").addEventListener("submit", async function (e) {
    e.preventDefault();
  
    // Collect form data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
  
    // Convert checkbox values to "Yes" or "No"
    data.International_plan = data.International_plan ? "Yes" : "No";
    data.Voice_mail_plan = data.Voice_mail_plan ? "Yes" : "No";
  
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
  
      const result = await response.json();
  
      if (response.ok) {
        const prediction = result.Prediction_Decision_Tree; // Use the prediction value
  
        // Animation for background color
        document.body.className = prediction === 1 ? "red" : "green";
        setTimeout(() => {
          document.body.className = "";
        }, 3000);
      } else {
        alert(result.detail || "An error occurred");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to get a prediction");
    }
  });
  