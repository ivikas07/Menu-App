document.addEventListener('DOMContentLoaded', () => {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const processedImage = document.getElementById('processed-image');

    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            const filter = button.getAttribute('data-filter');
            fetch('/apply_filter', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filter })
            })
            .then(response => response.json())
            .then(data => {
                processedImage.src = `data:image/jpeg;base64,${data.image}`;
            })
            .catch(error => console.error('Error:', error));
        });
    });
});
