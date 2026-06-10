document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("loan-form");
    const messageBox = document.getElementById("message-box");
    const submitButton = document.getElementById("predict-button");

    function clearMessages() {
        messageBox.innerHTML = "";
    }

    function showMessage(message, type = "success") {
        const card = document.createElement("div");
        card.className = `message ${type}`;
        card.innerHTML = `<p>${message}</p>`;
        messageBox.appendChild(card);
    }

    async function predictLoan(formData) {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(formData),
        });

        const contentType = response.headers.get("content-type") || "";
        let payload;

        if (contentType.includes("application/json")) {
            payload = await response.json();
        } else {
            const text = await response.text();
            throw new Error(text || "Prediction failed.");
        }

        if (!response.ok) {
            throw new Error(payload.error || "Prediction failed.");
        }

        return payload;
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        clearMessages();

        const formData = {};
        Array.from(form.elements).forEach((element) => {
            if (!element.name) {
                return;
            }
            formData[element.name] = element.value;
        });

        submitButton.disabled = true;
        submitButton.classList.add("loading");

        try {
            const payload = await predictLoan(formData);
            showMessage(`Prediction result: <strong>${payload.prediction}</strong>`, "success");
            if (payload.model_accuracy) {
                showMessage(`Model accuracy: ${payload.model_accuracy}%`, "success");
            }
        } catch (error) {
            showMessage(error.message, "error");
        } finally {
            submitButton.disabled = false;
            submitButton.classList.remove("loading");
        }
    });
});