document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const userInput = document.getElementById('user_input').value;

    const responseContainer = document.getElementById('response');
    const contextContainer = document.getElementById('context');

    responseContainer.innerHTML = "<p>Loading...</p>";
    contextContainer.innerHTML = "";

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `user_input=${encodeURIComponent(userInput)}`
        });

        if (response.ok) {
            const data = await response.json();
            responseContainer.innerHTML = `<p><strong>Response:</strong> ${data.answer}</p>`;
            data.context.forEach((doc, idx) => {
                contextContainer.innerHTML += `<p><strong>Document ${idx + 1}:</strong> ${doc}</p>`;
            });
        } else {
            const error = await response.json();
            responseContainer.innerHTML = `<p style="color: red;">Error: ${error.error}</p>`;
        }
    } catch (err) {
        responseContainer.innerHTML = `<p style="color: red;">An unexpected error occurred.</p>`;
    }
});
